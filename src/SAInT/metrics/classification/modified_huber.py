import torch
import numpy as np

# https://arxiv.org/pdf/2301.05579

def modified_huber_np(inp, targ):
    """Modified Huber Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    loss = np.where(z >= -1,
                    0.25 * (1 - z)**2,
                    -z)
    return np.mean(loss)

def modified_huber_torch(inp, targ):
    """Modified Huber Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    loss = torch.where(z >= -1,
                    0.25 * (1 - z)**2,
                    -z)
    return torch.mean(loss)

def modified_huber(inp, targ):
    """Modified Huber Loss
    # if z >= 1   then   1/4 * (1 - z)**2
    #             else   -z
    """
    if isinstance(inp, np.ndarray):
        return modified_huber_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return modified_huber_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
