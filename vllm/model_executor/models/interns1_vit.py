# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from collections.abc import Iterable
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.utils import torch_int

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               InternColumnParallelLinear,
                                               InternRowParallelLinear,
                                               RowParallelLinear,
                                               InternQKVParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': nn.LayerNorm,
}

class RMSNormTP(nn.Module):
    name = "RMSNormTP"

    def __init__(self, shard_sizes, eps) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.shard_sizes = shard_sizes
        self.weight = nn.Parameter(torch.ones(self.shard_sizes[self.tp_rank]))
        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps
        return

    def weight_loader(self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:

        shard_size = self.shard_sizes[self.tp_rank]
        shard = slice(sum(self.shard_sizes[:self.tp_rank]),sum(self.shard_sizes[:self.tp_rank]) + shard_size)
        param.data.copy_(loaded_weight[shard])
        return

class InternS1VisionPatchEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] //
                                                          patch_size[0])
        patch_shape = (image_size[0] // patch_size[0],
                       image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels,
                                    hidden_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values "
                "match with the one set in the configuration.")

        embeddings = self.projection(
            pixel_values.to(self.projection.weight.dtype))
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)


class InternS1VisionEmbeddings(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = InternS1VisionPatchEmbeddings(config)
        self.patch_size = config.patch_size
        self.image_size = (config.image_size if isinstance(
            config.image_size, Iterable) else
                           (config.image_size, config.image_size))
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int,
                                 width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """  # noqa: E501

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model
        # works for dynamic input shapes
        if not torch.jit.is_tracing(
        ) and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions,
                                                  sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        embeddings, (patch_height,
                     patch_width) = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width)

        return embeddings, (patch_height, patch_width)

def get_distribute_size(feature_size: int, world_size: int, rank: int, align: int = 1):
    assert feature_size % align == 0
    aligned_size = feature_size // align
    # try to make every rank has same amount of feats
    updated_aligned_size = aligned_size // world_size
    # if there are still some remain, given them to
    # each rank
    if rank < aligned_size % world_size:
        updated_aligned_size += 1
    return updated_aligned_size * align

class InternParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        num_dummy_heads: int = 0,
        prefix: str = ""
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim
        self.shard_heads = [get_distribute_size(self.num_heads, self.tp_size, rank) for rank in range(self.tp_size)]
        self.shard_sizes = [size_ * self.head_dim for size_ in self.shard_heads]
        self.num_heads_per_partition = self.shard_heads[self.tp_rank]

        self.q_size = self.shard_sizes[self.tp_rank]
        self.kv_size = self.shard_sizes[self.tp_rank]


        self.scale = self.head_dim**-0.5

        self.qkv_proj = InternQKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=config.attention_bias,
            #disable_tp=use_data_parallel,
            prefix=f"{prefix}.qkv_proj")

        self.qk_normalization = config.use_qk_norm
        if self.qk_normalization:
            self.q_norm = RMSNormTP(self.shard_sizes,
                                  eps=config.layer_norm_eps)
            self.k_norm = RMSNormTP(self.shard_sizes,
                                  eps=config.layer_norm_eps)

        self.projection_layer = InternRowParallelLinear(
            self.dummy_dim,
            self.embed_dim,
            self.shard_sizes,
            prefix=f"{prefix}.proj",
        )

    def qkv_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size == 1:
            q = self.q_norm(q).view(q_shape)
            k = self.k_norm(k).view(k_shape)
            return q, k

        dtype = q.dtype

        # variance
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        variance_q = (q * q).sum(-1, keepdim=True)
        variance_k = (k * k).sum(-1, keepdim=True)
        variance = torch.stack([variance_q, variance_k], dim=0)
        
        tensor_model_parallel_all_reduce(variance)

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        variance = variance / self.dummy_dim + self.config.layer_norm_eps
        variance_q, variance_k = variance
        q = q * torch.rsqrt(variance_q)
        q = q.to(dtype) * self.q_norm.weight
        k = k * torch.rsqrt(variance_k)
        k = k.to(dtype) * self.k_norm.weight

        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        

        if self.qk_normalization:
            q, k = self.qkv_norm(q, k)
        
        b, s, h = q.size()
        n = self.shard_heads[self.tp_rank]
        q = q.view(b, s, n, self.head_dim)
        k = k.view(b, s, n, self.head_dim)
        v = v.view(b, s, n, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        
        out = out.transpose(1, 2)
        out = out.flatten(-2, -1)

        out, _ = self.projection_layer(out)
        return out

class InternSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        num_dummy_heads: int = 0,
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim,
                                self.num_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(self.embed_dim,
                                self.num_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim,
                                self.num_heads * self.head_dim,
                                bias=config.attention_bias)

        self.qk_normalization = config.use_qk_norm
        if self.qk_normalization:
            self.q_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)
            self.k_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)

        self.projection_layer = nn.Linear(self.dummy_dim, self.embed_dim)

        # Use unified MultiHeadAttention with automatic backend selection
        self.attn = MultiHeadAttention(self.num_heads, self.head_dim,
                                       self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.qk_normalization:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Use unified MultiHeadAttention with automatic backend selection
        x = self.attn(q, k, v)

        x = self.projection_layer(x)
        return x


class InternS1VisionMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class InternS1VisionLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.attention = self._init_attn(config,
                                         quant_config,
                                         num_dummy_heads=num_dummy_heads,
                                         prefix=f"{prefix}.attention")

        self.mlp = InternS1VisionMLP(config,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")
        self.layernorm_before = NORM2FN[config.norm_type](
            config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = NORM2FN[config.norm_type](
            config.hidden_size, eps=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        self.lambda_1 = nn.Parameter(init_values *
                                     torch.ones(config.hidden_size),
                                     requires_grad=True)
        self.lambda_2 = nn.Parameter(init_values *
                                     torch.ones(config.hidden_size),
                                     requires_grad=True)

    def _init_attn(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        num_dummy_heads: int,
        prefix: str = "",
    ):
        import os
        if os.getenv("INTERNS1_VIT_USE_TP", "0") == "1":
            return InternParallelAttention(config,
                                       num_dummy_heads=num_dummy_heads,
                                       prefix=prefix)
        else:
            return InternSdpaAttention(config, num_dummy_heads=num_dummy_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = hidden_states + self.attention(
            self.layernorm_before(hidden_states)) * self.lambda_1

        hidden_states = hidden_states + self.mlp(
            self.layernorm_after(hidden_states)) * self.lambda_2

        return hidden_states


class InternS1VisionEncoder(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layer = nn.ModuleList([
            InternS1VisionLayer(config,
                                quant_config,
                                num_dummy_heads=num_dummy_heads,
                                prefix=f"{prefix}.layer.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor):

        hidden_states = inputs_embeds
        for encoder_layer in self.layer:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class InternS1VisionModel(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.embeddings = InternS1VisionEmbeddings(config)
        self.encoder = InternS1VisionEncoder(
            config=config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.encoder",
        )
        self.layernorm = None
        if not config.use_mean_pooling:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                'You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states, _ = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f'wrong pixel_values size: {pixel_values.shape}')

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        
        last_encoder_outputs = encoder_outputs
        
        if self.layernorm is not None:
            last_encoder_outputs = self.layernorm(encoder_outputs)

        return encoder_outputs, last_encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
