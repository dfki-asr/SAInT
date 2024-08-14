import torch
import numpy as np

def r2_np(inp, targ):
    """R-squared Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    ss_res = np.sum((targ - inp) ** 2)
    ss_tot = np.sum((targ - np.mean(targ)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    r2_loss = 1 - r2
    return r2_loss

def r2_torch(inp, targ):
    """R-squared Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    ss_res = torch.sum((targ - inp) ** 2)
    ss_tot = torch.sum((targ - torch.mean(targ)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    r2_loss = 1 - r2
    return r2_loss

def r2(inp, targ):
    """R-squared Loss"""
    if isinstance(inp, np.ndarray):
        return r2_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return r2_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
