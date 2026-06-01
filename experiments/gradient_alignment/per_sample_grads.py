"""Per-sample bias and weight gradient extraction.

Uses torch.func.vmap + torch.func.grad (PyTorch >= 2.0) with a loop fallback.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.func is available in PyTorch >= 2.0
from torch.func import functional_call, grad, vmap


# ---------------------------------------------------------------------------
# Layer enumeration helpers
# ---------------------------------------------------------------------------

def get_bias_bearing_layers(model: nn.Module) -> List[Tuple[int, str, nn.Module]]:
    """Return (index, dotted_name, module) for all bias-bearing layers, sorted
    in the order they appear in model.named_modules().

    Instruments nn.Conv2d, nn.Linear, nn.BatchNorm2d/1d with affine=True
    (and any other module where module.bias is a non-None Parameter).
    """
    result = []
    idx = 0
    for name, module in model.named_modules():
        if not name:
            continue
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter) and bias is not None:
            result.append((idx, name, module))
            idx += 1
    return result


def get_weight_bearing_layers(model: nn.Module) -> List[Tuple[int, str, nn.Module]]:
    """Same as get_bias_bearing_layers but for weight parameters."""
    result = []
    idx = 0
    for name, module in model.named_modules():
        if not name:
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, nn.Parameter) and weight is not None:
            result.append((idx, name, module))
            idx += 1
    return result


# ---------------------------------------------------------------------------
# Per-sample gradient extraction
# ---------------------------------------------------------------------------

def per_sample_bias_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    target_label: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample bias gradients for all bias-bearing layers.

    Args:
        model:        Eval-mode model on ``device``.
        inputs:       Normalised batch (N, C, H, W) on ``device``.
        target_label: Backdoor target label (integer).
        device:       Torch device.

    Returns:
        Dict mapping layer name → tensor of shape (N, bias_dim),
        one gradient row per sample.
    """
    model.eval()
    N = inputs.shape[0]
    targets = torch.full((N,), target_label, dtype=torch.long, device=device)

    try:
        return _per_sample_grads_vmap(model, inputs, targets, param_type="bias")
    except Exception as exc:
        print(f"[per_sample_grads] vmap failed ({exc}); using loop fallback.")
        return _per_sample_grads_loop(model, inputs, targets, param_type="bias")


def per_sample_bias_gradients_from_labels(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Per-sample bias gradients using each sample's own true label.

    Identical to :func:`per_sample_bias_gradients` except ``labels`` is a
    ``(N,)`` long tensor of per-sample class indices rather than a single
    integer target.  Used for computing the *clean* gradient covariance
    Σ_clean = (1/N) Σ_i g_i g_i^T.

    Args:
        model:   Eval-mode model on ``device``.
        inputs:  Normalised batch ``(N, C, H, W)`` on ``device``.
        labels:  True class labels ``(N,)`` long tensor.
        device:  Torch device.

    Returns:
        Dict mapping layer name → tensor of shape ``(N, bias_dim)``.
    """
    model.eval()
    targets = labels.to(device)
    try:
        return _per_sample_grads_vmap(model, inputs, targets, param_type="bias")
    except Exception as exc:
        print(f"[per_sample_grads] vmap failed ({exc}); using loop fallback.")
        return _per_sample_grads_loop(model, inputs, targets, param_type="bias")


def per_sample_weight_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    target_label: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Same as per_sample_bias_gradients but returns flattened weight grads.

    Returns:
        Dict mapping layer name → tensor of shape (N, weight_numel).
    """
    model.eval()
    N = inputs.shape[0]
    targets = torch.full((N,), target_label, dtype=torch.long, device=device)

    try:
        return _per_sample_grads_vmap(model, inputs, targets, param_type="weight")
    except Exception as exc:
        print(f"[per_sample_grads] vmap failed ({exc}); using loop fallback.")
        return _per_sample_grads_loop(model, inputs, targets, param_type="weight")


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------

def _per_sample_grads_vmap(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    param_type: str,   # "bias" or "weight"
) -> Dict[str, torch.Tensor]:
    """Implementation using torch.func.vmap + torch.func.grad."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_single(params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0), reduction="sum")

    grad_fn = grad(loss_single, argnums=0)

    # vmap over the batch — neither params nor buffers are batched
    per_grads: Dict[str, torch.Tensor] = vmap(
        lambda x, y: grad_fn(params, x, y),
        in_dims=(0, 0),
        randomness="different",
    )(inputs, targets)
    # per_grads: {param_name: (N, *param_shape)}

    # Filter to the requested param type and reshape weight grads to 2-D
    result: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        param = getattr(module, param_type, None)
        if not (isinstance(param, nn.Parameter) and param is not None):
            continue
        key = f"{name}.{param_type}"
        if key not in per_grads:
            continue
        g = per_grads[key]        # (N, *param_shape)
        if param_type == "weight":
            g = g.reshape(g.shape[0], -1)    # (N, weight_numel)
        result[name] = g

    return result


def _per_sample_grads_loop(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    param_type: str,
) -> Dict[str, torch.Tensor]:
    """Fallback: iterate over individual samples."""
    N = inputs.shape[0]
    accum: Dict[str, List[torch.Tensor]] = {}

    for i in range(N):
        model.zero_grad()
        x = inputs[i : i + 1]
        y = targets[i : i + 1]
        out = model(x)
        F.cross_entropy(out, y, reduction="sum").backward()

        for name, module in model.named_modules():
            if not name:
                continue
            param = getattr(module, param_type, None)
            if not (isinstance(param, nn.Parameter) and param is not None):
                continue
            g = param.grad
            if g is None:
                continue
            g = g.detach().clone()
            if param_type == "weight":
                g = g.reshape(-1)
            accum.setdefault(name, []).append(g)

    model.zero_grad()
    return {name: torch.stack(grads, dim=0) for name, grads in accum.items()}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    target_label: int,
    device: torch.device,
) -> bool:
    """Verify that mean of per-sample bias grads matches the batch grad.

    Prints PASS / FAIL with the max absolute error for each layer.
    Returns True if all layers pass.
    """
    model.eval()
    N = inputs.shape[0]
    targets = torch.full((N,), target_label, dtype=torch.long, device=device)

    # --- per-sample approach -------------------------------------------------
    per_sample = _per_sample_grads_vmap(model, inputs, targets, param_type="bias")
    per_sample_mean = {name: g.mean(dim=0) for name, g in per_sample.items()}

    # --- batch approach (standard .backward()) --------------------------------
    model.zero_grad()
    out = model(inputs)
    loss = F.cross_entropy(out, targets, reduction="mean")
    loss.backward()

    all_pass = True
    for name, module in model.named_modules():
        if not name:
            continue
        bias = getattr(module, "bias", None)
        if not (isinstance(bias, nn.Parameter) and bias is not None):
            continue
        if name not in per_sample_mean:
            continue
        batch_grad = bias.grad
        if batch_grad is None:
            continue
        max_err = (batch_grad - per_sample_mean[name]).abs().max().item()
        status = "PASS" if max_err < 1e-4 else "FAIL"
        if max_err >= 1e-4:
            all_pass = False
        print(f"  Sanity [{status}] layer={name:40s}  max_err={max_err:.2e}")

    model.zero_grad()
    return all_pass


# ---------------------------------------------------------------------------
# α computation helpers
# ---------------------------------------------------------------------------

def compute_alpha(
    per_sample_grads: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute per-sample cosine alignment α_i^(l) for each layer.

    For each layer l:
        μ^(l) = mean over i of g_i^(l)
        u^(l) = μ^(l) / ||μ^(l)||
        α_i^(l) = dot( g_i^(l)/||g_i^(l)||, u^(l) )

    Args:
        per_sample_grads: {layer_name: (N, dim)} from per_sample_*_gradients.

    Returns:
        {layer_name: (N,)} alpha values in [-1, 1].
    """
    result: Dict[str, torch.Tensor] = {}
    for name, g in per_sample_grads.items():
        N = g.shape[0]
        if N == 0 or g.shape[1] == 0:
            continue

        mu = g.mean(dim=0)                          # (dim,)
        mu_norm = mu.norm(p=2)
        if mu_norm < 1e-12:
            result[name] = torch.zeros(N)
            continue

        u = mu / mu_norm                            # unit mean direction

        g_norms = g.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        g_unit = g / g_norms                        # (N, dim)

        alpha = (g_unit @ u)                        # (N,)
        result[name] = alpha.cpu()

    return result
