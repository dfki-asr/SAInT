import torch
import numpy as np

# https://arxiv.org/pdf/2301.05579

def hinge_np(inp, targ):
    """Hinge Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return np.mean(np.maximum(0, 1 - inp * targ))

def hinge_torch(inp, targ):
    """Hinge Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return torch.mean(torch.clamp(1 - inp * targ, min=0))

def hinge(inp, targ):
    """Hinge Loss"""
    if isinstance(inp, np.ndarray):
        return hinge_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return hinge_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
