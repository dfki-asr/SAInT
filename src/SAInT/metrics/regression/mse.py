import torch
import numpy as np

def mse_np(inp, targ):
    """Mean Squared Error (MSE): NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return np.mean((inp - targ) ** 2)

def mse_torch(inp, targ):
    """Mean Squared Error (MSE): PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    return torch.mean((inp - targ) ** 2)

def mse(inp, targ):
    """Mean Squared Error (MSE)"""
    if isinstance(inp, np.ndarray):
        return mse_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return mse_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
