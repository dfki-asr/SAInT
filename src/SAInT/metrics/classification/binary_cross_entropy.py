import torch
import numpy as np

def binary_cross_entropy_np(preds, targets):
    """Binary Cross-Entropy Loss (BCE): NumPy implementation"""
    assert preds.shape == targets.shape, "Shapes of predictions and targets must match"
    eps = 1e-15  # Small value to avoid log(0)
    preds = np.clip(preds, eps, 1 - eps)  # Avoid log(0) by clipping probabilities
    # Compute Binary Cross-Entropy Loss
    loss = -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
    return loss

def binary_cross_entropy_torch(preds, targets):
    """Binary Cross-Entropy Loss (BCE): PyTorch implementation"""
    assert preds.shape == targets.shape, "Shapes of predictions and targets must match"
    eps = 1e-15  # Small value to avoid log(0)
    preds = torch.clamp(preds, eps, 1 - eps)  # Avoid log(0) by clipping probabilities
    # Compute Binary Cross-Entropy Loss
    loss = -torch.mean(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
    return loss

def binary_cross_entropy(inp, targ):
    """Binary Cross-Entropy Loss (BCE)"""
    if isinstance(inp, np.ndarray) and isinstance(targ, np.ndarray):
        return binary_cross_entropy_np(inp, targ)
    elif isinstance(inp, torch.Tensor) and isinstance(targ, torch.Tensor):
        return binary_cross_entropy_torch(inp, targ)
    else:
        raise TypeError("Inputs should be either NumPy arrays or PyTorch tensors.")
