# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers.configuration_utils import ALLOWED_LAYER_TYPES, PretrainedConfig


class Step3p5Config(PretrainedConfig):
    model_type = "step3p5"

    def validate_layer_type(self) -> None:
        """HF assumes ``len(layer_types) == num_hidden_layers``; Step3.5 MTP checkpoints
        append ``num_nextn_predict_layers`` extra entries for MTP blocks."""
        if not (
            getattr(self, "layer_types", None) is not None
            and hasattr(self, "num_hidden_layers")
        ):
            return
        if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in self.layer_types):
            raise ValueError(
                f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES} "
                f"but got {self.layer_types}"
            )
        n = self.num_hidden_layers
        if n is None:
            return
        num_mtp = int(getattr(self, "num_nextn_predict_layers", 0) or 0)
        length = len(self.layer_types)
        if length == n:
            return
        if num_mtp > 0 and length == n + num_mtp:
            return
        raise ValueError(
            f"`num_hidden_layers` ({n}) must match len(layer_types) ({length}), "
            f"or with num_nextn_predict_layers={num_mtp} expect length {n + num_mtp}."
        )

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        rms_norm_eps: float = 1e-5,
        moe_every_n_layer: int = 2,
        use_moe: bool = False,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        moe_layer_offset: int = 0,
        rope_theta: float | list[float] | None = 500000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        share_expert_dim: int | None = None,
        norm_expert_weight: bool = True,
        bos_token_id: list[int] | int | None = None,
        eos_token_id: list[int] | int | None = None,
        moe_router_activation: str = "softmax",
        moe_router_scaling_factor: float = 1.0,
        att_impl_type: str = "GQA",
        use_head_wise_attn_gate: bool = False,
        use_moe_router_bias: bool = True,
        need_fp32_gate: bool = True,
        layer_types: list[str] | None = None,
        use_rope_layers: list[bool] | None = None,
        yarn_only_types: list[str] | None = None,
        attention_other_setting: dict[str, Any] | None = None,
        num_nextn_predict_layers: int = 0,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        max_position_embeddings: int | None = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_every_n_layer = moe_every_n_layer
        self.moe_num_experts = moe_num_experts
        self.num_experts_per_tok = moe_top_k
        self.moe_top_k = moe_top_k
        self.moe_layer_offset = moe_layer_offset

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        if share_expert_dim is None:
            self.share_expert_dim = self.moe_intermediate_size * self.moe_top_k
        else:
            self.share_expert_dim = share_expert_dim
        self.norm_expert_weight = norm_expert_weight

        self.max_position_embeddings = max_position_embeddings
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate

        self.att_impl_type = att_impl_type
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        # Checkpoints may append MTP block entries after the main transformer layers:
        # indices [0, num_hidden_layers) are backbone; [num_hidden_layers,
        # num_hidden_layers + num_nextn_predict_layers) are MTP (see step3p5_mtp.py).
        # Only strip trailing junk if the list is longer than that logical span.
        layer_types_extended_for_restore: list[str] | None = None
        if layer_types is not None:
            max_entries = num_hidden_layers + num_nextn_predict_layers
            if len(layer_types) > max_entries:
                layer_types = layer_types[:max_entries]
            num_mtp = num_nextn_predict_layers
            if (
                num_mtp > 0
                and len(layer_types) == num_hidden_layers + num_mtp
            ):
                # PreTrainedConfig.validate runs inside super().__init__ and always uses
                # the parent's validate_layer_type (expects len(layer_types)==num_hidden_layers).
                # Temporarily expose only backbone entries during super(); restore below.
                layer_types_extended_for_restore = list(layer_types)
                self.layer_types = layer_types[:num_hidden_layers]
            else:
                self.layer_types = layer_types
        else:
            self.layer_types = None
        self.use_rope_layers = use_rope_layers
        self.yarn_only_types = yarn_only_types
        self.attention_other_setting = attention_other_setting
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared

        resolved_bos_token_id = 1 if bos_token_id is None else bos_token_id
        resolved_eos_token_id = [2, 3] if eos_token_id is None else eos_token_id
        self.bos_token_id = resolved_bos_token_id
        self.eos_token_id = resolved_eos_token_id

        super().__init__(
            bos_token_id=resolved_bos_token_id,
            eos_token_id=resolved_eos_token_id,
            **kwargs,
        )
        if layer_types_extended_for_restore is not None:
            self.layer_types = layer_types_extended_for_restore

# `huggingface_hub` @strict on `PreTrainedConfig` copies `__class_validators__` from
# the parent, so `validate_layer_type` still points at the base implementation and
# never calls the override above. Replace that entry for this model class only.
def _register_step3p5_layer_type_validator() -> None:
    parents = getattr(Step3p5Config, "__class_validators__", None)
    if not parents:
        return
    Step3p5Config.__class_validators__ = tuple(  # type: ignore[misc]
        Step3p5Config.validate_layer_type
        if getattr(v, "__name__", None) == "validate_layer_type"
        else v
        for v in parents
    )


_register_step3p5_layer_type_validator()
