import torch
import numpy as np

def rmse_np(inp, targ):
    """Root Mean Squared Error (RMSE): NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return np.sqrt(np.mean((inp - targ) ** 2))

def rmse_torch(inp, targ):
    """Root Mean Squared Error (RMSE): PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return torch.sqrt(torch.mean((inp - targ) ** 2))

def rmse(inp, targ):
    """Root Mean Squared Error (RMSE)"""
    if isinstance(inp, np.ndarray):
        return rmse_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return rmse_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
