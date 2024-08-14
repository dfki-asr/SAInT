import torch
import numpy as np

# Sources:
# https://arxiv.org/pdf/2301.05579,
# http://qwone.com/~jason/writing/smoothHinge.pdf

def smoothed_hinge_np(inp, targ):
    """Smoothed Hinge Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    loss = np.where(z >= 1, 0, np.where(z > 0, 0.5 * (1 - z)**2, 0.5 - z))
    return np.mean(loss)

def smoothed_hinge_torch(inp, targ):
    """Smoothed Hinge Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    loss = torch.where(z >= 1, torch.tensor(0.0), torch.where(z > 0, 0.5 * (1 - z)**2, 0.5 - z))
    return torch.mean(loss)

def smoothed_hinge(inp, targ):
    """Smoothed Hinge Loss
    For z ≥ 1       then 0             correct classification with a margin z of at least 1
    For 0 < z < 1   then 0.5 (1-z)**2  loss is a quadratic term for smooth transition
    For z ≤ 0:      then 0.5 - z       similar to the traditional hinge loss, but it is linear
    """
    if isinstance(inp, np.ndarray):
        return smoothed_hinge_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return smoothed_hinge_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
