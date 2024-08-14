import torch
import numpy as np

def cross_entropy_np(inp, targ):
    """Cross-Entropy Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of predictions and targets must match"
    eps = 1e-15  # Small value to avoid log(0)
    inp = np.clip(inp, eps, 1 - eps)  # Avoid log(0) by clipping probabilities
    # Compute Cross-Entropy Loss
    loss = -np.sum(targ * np.log(inp)) / inp.shape[0]
    return loss

def cross_entropy_torch(inp, targ):
    """Cross-Entropy Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of predictions and targets must match"
    eps = 1e-15  # Small value to avoid log(0)
    inp = torch.clamp(inp, eps, 1 - eps)  # Avoid log(0) by clipping probabilities
    # Compute Cross-Entropy Loss
    loss = -torch.sum(targ * torch.log(inp)) / inp.shape[0]
    return loss

def cross_entropy(inp, targ):
    """Cross-Entropy Loss"""
    if isinstance(inp, np.ndarray) and isinstance(targ, np.ndarray):
        return cross_entropy_np(inp, targ)
    elif isinstance(inp, torch.Tensor) and isinstance(targ, torch.Tensor):
        return cross_entropy_torch(inp, targ)
    else:
        raise TypeError("Inputs should be either NumPy arrays or PyTorch tensors.")
