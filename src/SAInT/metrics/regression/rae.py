import torch
import numpy as np

def rae_np(inp, targ):
    """Relative Absolute Error (RAE): NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    numerator = np.sum(np.abs(inp - targ))
    denominator = np.sum(np.abs(targ - np.mean(targ)))
    return numerator / denominator

def rae_torch(inp, targ):
    """Relative Absolute Error (RAE): PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    numerator = torch.sum(torch.abs(inp - targ))
    denominator = torch.sum(torch.abs(targ - torch.mean(targ)))
    return numerator / denominator

def rae(inp, targ):
    """Relative Absolute Error (RAE)"""
    if isinstance(inp, np.ndarray):
        return rae_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return rae_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
