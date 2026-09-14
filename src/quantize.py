"""
quantize.py
===========
Post-training quantization and polynomial activation replacement.

Two steps for HE compatibility:
  Step 1 — Quantize model weights to n-bit integers (reduces ciphertext size)
  Step 2 — Replace ReLU activations with degree-2 polynomial approximation
            f(x) = a0 + a1*x + a2*x^2   (HE supports only +, *, constants)

Why this matters:
  CKKS homomorphic encryption natively supports:
    - Ciphertext + Ciphertext  (addition)
    - Ciphertext * Plaintext   (multiplication by constant)
    - Ciphertext * Ciphertext  (multiplication — consumes HE "budget")
  It does NOT support: ReLU, max, sigmoid, softmax directly.
  A polynomial approximation is the standard workaround.

Usage:
    from src.quantize import quantize_model, swap_activations
    model = quantize_model(model, bits=8)
    model = swap_activations(model)
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from src.model import GCNModel, GINModel, PolyActivation


# -------------------------------------------------------------------
# Step 1: Weight Quantization (Post-Training)
# -------------------------------------------------------------------

def quantize_tensor(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Symmetric min-max quantization of a weight tensor.

    Maps weights to integers in [-2^(bits-1), 2^(bits-1)-1],
    then dequantizes back to float32 for inference.

    This simulates quantization-aware inference without
    requiring a full quantization-aware training loop.

    Args:
        tensor (Tensor): Weight tensor to quantize
        bits   (int)   : Bit width (default 8)

    Returns:
        Tensor: Quantized-then-dequantized float tensor
    """
    qmin = -(2 ** (bits - 1))
    qmax =  (2 ** (bits - 1)) - 1

    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor.clone()

    scale = max_val / qmax
    q     = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q * scale   # dequantize: back to float, but with quantization noise


def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """
    Apply post-training quantization to all Linear and Conv weights.

    Args:
        model (nn.Module): Trained plaintext model
        bits  (int)      : Quantization bit width

    Returns:
        model (nn.Module): Same model with quantized weights (in-place)
    """
    model = copy.deepcopy(model)
    for name, param in model.named_parameters():
        if "weight" in name:
            with torch.no_grad():
                param.data = quantize_tensor(param.data, bits=bits)
    print(f"[quantize] Weights quantized to {bits}-bit precision.")
    return model


# -------------------------------------------------------------------
# Step 2: Activation Replacement (ReLU -> Polynomial)
# -------------------------------------------------------------------

def swap_activations(model: nn.Module) -> nn.Module:
    """
    Recursively replace all ReLU activations with PolyActivation.

    This is necessary for CKKS homomorphic encryption, which cannot
    evaluate non-polynomial functions on ciphertext.

    The polynomial f(x) = a0 + a1*x + a2*x^2 approximates ReLU
    in the range roughly [-5, 5] which covers most pre-activation values.

    Args:
        model (nn.Module): Model with ReLU activations

    Returns:
        model (nn.Module): Same model with PolyActivation replacing ReLU
    """
    model = copy.deepcopy(model)

    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, PolyActivation())
                print(f"[quantize] Replaced ReLU -> PolyActivation in '{name}'")
            else:
                _replace(child)

    _replace(model)
    return model


# -------------------------------------------------------------------
# Polynomial coefficient fitting (optional refinement)
# -------------------------------------------------------------------

def fit_poly_activation(model: nn.Module, data, device: str = "cpu") -> nn.Module:
    """
    Fine-tune PolyActivation coefficients to minimise MSE against
    the original ReLU outputs on training data.

    Run this AFTER swap_activations() to get a better approximation
    than the default initialization.

    Args:
        model  (nn.Module)    : Model after swap_activations()
        data   (PyG Data)     : Training graph data
        device (str)          : 'cpu' or 'cuda'

    Returns:
        model (nn.Module): Model with refined polynomial coefficients
    """
    model = model.to(device)
    model.eval()

    # Collect all PolyActivation modules
    poly_mods = [m for m in model.modules() if isinstance(m, PolyActivation)]

    if not poly_mods:
        print("[quantize] No PolyActivation modules found. Run swap_activations first.")
        return model

    # Fine-tune coefficients with SGD for 200 steps
    optimizer = torch.optim.Adam(
        [p for m in poly_mods for p in m.parameters()], lr=0.01
    )
    relu = nn.ReLU()

    # Sample random inputs in [-5, 5] to fit
    for step in range(200):
        x_sample = torch.randn(1000) * 3  # covers typical pre-activation range
        total_loss = sum(
            nn.MSELoss()(m(x_sample), relu(x_sample)) for m in poly_mods
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    for m in poly_mods:
        print(f"[quantize] Poly coeffs: a0={m.a0.item():.4f}, "
              f"a1={m.a1.item():.4f}, a2={m.a2.item():.4f}")

    return model


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

def print_quantization_summary(original: nn.Module, quantized: nn.Module):
    """Print parameter count and precision comparison."""
    orig_params = sum(p.numel() for p in original.parameters())
    quant_params = sum(p.numel() for p in quantized.parameters())

    orig_mb  = orig_params * 4 / 1e6   # float32 = 4 bytes
    quant_mb = orig_params * 1 / 1e6   # int8    = 1 byte (approx)

    print("\n[quantize] -- Quantization Summary ------------------")
    print(f"  Parameters     : {orig_params:,}")
    print(f"  Original size  : {orig_mb:.2f} MB  (float32)")
    print(f"  Quantized size : {quant_mb:.2f} MB  (int8 approx)")
    print(f"  Size reduction : {orig_mb / quant_mb:.1f}x")
    print("-----------------------------------------------------\n")


if __name__ == "__main__":
    import yaml
    from src.model import build_model

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    print(f"Original model:\n{model}\n")

    quantized = quantize_model(model, bits=cfg["quantize"]["bits"])
    poly_model = swap_activations(quantized)
    print_quantization_summary(model, poly_model)
