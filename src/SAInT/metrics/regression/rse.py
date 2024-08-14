import torch
import numpy as np

def rse_np(inp, targ):
    """Relative Squared Error (RSE): NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    numerator = np.sum((inp - targ) ** 2)
    denominator = np.sum((targ - np.mean(targ)) ** 2)
    return numerator / denominator

def rse_torch(inp, targ):
    """Relative Squared Error (RSE): PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    numerator = torch.sum((inp - targ) ** 2)
    denominator = torch.sum((targ - torch.mean(targ)) ** 2)
    return numerator / denominator

def rse(inp, targ):
    """Relative Squared Error (RSE)"""
    if isinstance(inp, np.ndarray):
        return rse_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return rse_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
