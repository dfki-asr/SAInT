import torch
import numpy as np

def rmsle_np(inp, targ):
    """Root Mean Squared Log Error (RMSLE): NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    # Ensure values are within valid range to prevent log(0) and negative values
    min_value = 1e-15
    inp = np.clip(inp, min_value, None)
    targ = np.clip(targ, min_value, None)
    log_inp = np.log(inp + 1)
    log_targ = np.log(targ + 1)
    return np.sqrt(np.mean((log_inp - log_targ) ** 2))

def rmsle_torch(inp, targ):
    """Root Mean Squared Log Error (RMSLE): PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    # Ensure values are within valid range to prevent log(0) and negative values
    min_value = 1e-15
    inp = torch.clamp(inp, min=min_value)
    targ = torch.clamp(targ, min=min_value)
    log_inp = torch.log(inp + 1)
    log_targ = torch.log(targ + 1)
    return torch.sqrt(torch.mean((log_inp - log_targ) ** 2))

def rmsle(inp, targ):
    """Root Mean Squared Log Error (RMSLE)"""
    if isinstance(inp, np.ndarray):
        return rmsle_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return rmsle_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
