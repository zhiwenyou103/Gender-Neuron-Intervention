"""Utilities for collecting and intervening on vLLM MLP activations."""

from __future__ import annotations

from types import MethodType
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from gender_neuron_utils import is_llama_like


def get_underlying_model(llm) -> object:
    """Return the wrapped torch model from vLLM V0 internals."""
    candidate_paths = (
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "driver_worker", "model_runner", "model"),
    )
    for path in candidate_paths:
        obj = llm
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not access the vLLM V0 model internals. "
        "Use vllm==0.10.1 with VLLM_USE_V1=0."
    )


def get_hf_config(llm) -> object:
    return llm.llm_engine.model_config.hf_config


def get_architecture_dimensions(llm, model_name: str) -> tuple[int, int]:
    config = get_hf_config(llm)
    num_layers = int(config.num_hidden_layers)
    if is_llama_like(model_name):
        intermediate_size = int(config.intermediate_size)
    else:
        intermediate_size = int(config.hidden_size) * 4
    return num_layers, intermediate_size


def attach_activation_collectors(
    llm,
    model_name: str,
    stats: Dict[str, torch.Tensor],
) -> List[object]:
    """Patch each MLP forward pass and collect product/pre-down-projection stats."""
    base_model = get_underlying_model(llm)
    llama_like = is_llama_like(model_name)
    num_layers = stats["sum1"].size(0)
    originals: List[object] = []
    for layer_idx in range(num_layers):
        mlp = get_mlp_module(base_model, layer_idx, llama_like)
        originals.append(mlp.forward)
        mlp.forward = MethodType(
            make_activation_collector_forward(layer_idx, stats, llama_like),
            mlp,
        )
    return originals


def attach_mlp_masks(
    llm,
    model_name: str,
    layer_zero_masks: List[torch.Tensor],
    layer_boost_masks: Optional[List[torch.Tensor]] = None,
    mask_factor: float = 1.0,
    boost_factor: float = 0.0,
    mask_at_product: bool = False,
) -> List[Optional[object]]:
    """Patch MLPs to suppress selected neurons and optionally boost kept neurons."""
    base_model = get_underlying_model(llm)
    llama_like = is_llama_like(model_name)
    originals: List[Optional[object]] = []

    for layer_idx, zero_mask in enumerate(layer_zero_masks):
        boost_mask = (
            layer_boost_masks[layer_idx]
            if layer_boost_masks is not None
            else torch.tensor([], dtype=torch.long)
        )
        if _is_empty(zero_mask) and _is_empty(boost_mask):
            originals.append(None)
            continue

        mlp = get_mlp_module(base_model, layer_idx, llama_like)
        originals.append(mlp.forward)
        mlp.forward = MethodType(
            make_masked_forward(
                _to_cuda_long(zero_mask),
                _to_cuda_long(boost_mask),
                llama_like,
                mask_factor,
                boost_factor,
                mask_at_product,
            ),
            mlp,
        )

    return originals


def restore_mlp_forwards(llm, model_name: str, originals: Iterable[Optional[object]]) -> None:
    base_model = get_underlying_model(llm)
    llama_like = is_llama_like(model_name)
    for layer_idx, original in enumerate(originals):
        if original is None:
            continue
        get_mlp_module(base_model, layer_idx, llama_like).forward = original


def get_mlp_module(base_model: object, layer_idx: int, llama_like: bool) -> object:
    if llama_like:
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            return base_model.model.layers[layer_idx].mlp
        if hasattr(base_model, "layers"):
            return base_model.layers[layer_idx].mlp
    else:
        if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            return base_model.transformer.h[layer_idx].mlp
    raise RuntimeError(f"Could not locate MLP module for layer {layer_idx}.")


def make_activation_collector_forward(
    layer_idx: int,
    stats: Dict[str, torch.Tensor],
    llama_like: bool,
):
    def llama_forward(self, x):
        if hasattr(self, "gate_up_proj"):
            gate_up = _unwrap_tuple(self.gate_up_proj(x))
            hidden = gate_up.size(-1) // 2
            activation = F.silu(gate_up[..., :hidden]) * gate_up[..., hidden:]
            _accumulate_activation(stats, layer_idx, activation)
            return _unwrap_tuple(self.down_proj(activation))

        if hasattr(self, "gate_proj") and hasattr(self, "up_proj"):
            gate = _unwrap_tuple(self.gate_proj(x))
            up = _unwrap_tuple(self.up_proj(x))
            activation = F.silu(gate) * up
            _accumulate_activation(stats, layer_idx, activation)
            return _unwrap_tuple(self.down_proj(activation))

        return bloom_forward(self, x)

    def bloom_forward(self, x):
        activation = _unwrap_tuple(self.dense_h_to_4h(x))
        activation = self.gelu_impl(activation) if hasattr(self, "gelu_impl") else F.gelu(activation)
        _accumulate_activation(stats, layer_idx, activation)
        return _unwrap_tuple(self.dense_4h_to_h(activation))

    return llama_forward if llama_like else bloom_forward


def make_masked_forward(
    zero_mask: torch.Tensor,
    boost_mask: torch.Tensor,
    llama_like: bool,
    mask_factor: float,
    boost_factor: float,
    mask_at_product: bool,
):
    def llama_forward(self, x):
        if hasattr(self, "gate_up_proj"):
            gate_up = _unwrap_tuple(self.gate_up_proj(x))
            hidden = gate_up.size(-1) // 2
            act = F.silu(gate_up[..., :hidden])
            up = gate_up[..., hidden:]
            if mask_at_product:
                y = _apply_masks(act * up, zero_mask, boost_mask, mask_factor, boost_factor)
            else:
                y = _apply_masks(act, zero_mask, boost_mask, mask_factor, boost_factor) * up
            return _unwrap_tuple(self.down_proj(y))

        if hasattr(self, "gate_proj") and hasattr(self, "up_proj"):
            gate = _unwrap_tuple(self.gate_proj(x))
            up = _unwrap_tuple(self.up_proj(x))
            act = F.silu(gate)
            if mask_at_product:
                y = _apply_masks(act * up, zero_mask, boost_mask, mask_factor, boost_factor)
            else:
                y = _apply_masks(act, zero_mask, boost_mask, mask_factor, boost_factor) * up
            return _unwrap_tuple(self.down_proj(y))

        return bloom_forward(self, x)

    def bloom_forward(self, x):
        y = _unwrap_tuple(self.dense_h_to_4h(x))
        y = self.gelu_impl(y) if hasattr(self, "gelu_impl") else F.gelu(y)
        y = _apply_masks(y, zero_mask, boost_mask, mask_factor, boost_factor)
        return _unwrap_tuple(self.dense_4h_to_h(y))

    return llama_forward if llama_like else bloom_forward


def _accumulate_activation(stats: Dict[str, torch.Tensor], layer_idx: int, activation: torch.Tensor) -> None:
    flat = activation.detach().float().reshape(-1, activation.size(-1))
    stats["sum1"][layer_idx].add_(flat.sum(dim=0))
    stats["sum2"][layer_idx].add_(flat.square().sum(dim=0))
    stats["over_zero"][layer_idx].add_((flat > 0).sum(dim=0).to(stats["over_zero"].dtype))


def _apply_masks(
    tensor: torch.Tensor,
    zero_mask: torch.Tensor,
    boost_mask: torch.Tensor,
    mask_factor: float,
    boost_factor: float,
) -> torch.Tensor:
    if zero_mask.numel() > 0:
        if mask_factor >= 1.0:
            tensor = tensor.index_fill(-1, zero_mask, 0)
        else:
            scale = torch.ones_like(tensor)
            scale = scale.index_fill(-1, zero_mask, 1.0 - mask_factor)
            tensor = tensor * scale

    if boost_factor > 0.0 and boost_mask.numel() > 0:
        scale = torch.ones_like(tensor)
        scale = scale.index_fill(-1, boost_mask, 1.0 + boost_factor)
        tensor = tensor * scale

    return tensor


def _unwrap_tuple(value):
    return value[0] if isinstance(value, tuple) else value


def _is_empty(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is None or len(tensor) == 0


def _to_cuda_long(tensor: Optional[torch.Tensor]) -> torch.Tensor:
    if tensor is None or len(tensor) == 0:
        return torch.tensor([], dtype=torch.long, device="cuda")
    return tensor.to(device="cuda", dtype=torch.long)
