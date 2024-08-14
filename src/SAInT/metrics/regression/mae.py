import torch
import numpy as np

def mae_np(inp, targ):
    """Mean Absolute Error (MAE): NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return np.mean(np.abs(inp - targ))

def mae_torch(inp, targ):
    """Mean Absolute Error (MAE): PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return torch.mean(torch.abs(inp - targ))

def mae(inp, targ):
    """Mean Absolute Error (MAE)"""
    if isinstance(inp, np.ndarray):
        return mae_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return mae_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
