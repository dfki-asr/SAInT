import torch
import numpy as np

# https://arxiv.org/pdf/2301.05579

def ramp_np(inp, targ):
    """Ramp Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    hinge = np.maximum(0, 1 - z)
    loss = np.where(z <= 1,
                    hinge,
                    1)
    return np.mean(loss)

def ramp_torch(inp, targ):
    """Ramp Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    hinge = torch.maximum(torch.tensor(0.0), 1 - z)
    loss = torch.where(z <= 1,
                    hinge,
                    1)
    return torch.mean(loss)

def ramp(inp, targ):
    """Ramp Loss
    # if z <= 1   then   hinge
    #             else   1
    """
    if isinstance(inp, np.ndarray):
        return ramp_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return ramp_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
