import torch
import numpy as np

# https://arxiv.org/pdf/2301.05579

def squared_hinge_np(inp, targ, gamma = 0.2):
    """Quadratically Smoothed (Squared) Hinge Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    g = np.array(gamma)
    loss = np.where(z >= 1 - g,
                    1 / (2 * g) * np.maximum(0, -z),
                    1 - 0.5 * g - z)
    return np.mean(loss)

def squared_hinge_torch(inp, targ,  gamma = 0.2):
    """Quadratically Smoothed (Squared) Hinge Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    z = inp * targ
    g = torch.tensor(gamma)
    loss = torch.where(z >= 1 - g,
                       1 / (2 * g) * torch.maximum(torch.tensor(0.0), -z),
                       1 - 0.5 * g - z)
    return torch.mean(loss)

def squared_hinge(inp, targ):
    """Quadratically Smoothed (Squared) Hinge Loss
    # if z >= 1 - gamma   then   1/(2*gamma) * max(0, -z)
    #                     else   1 -  0.5 * gamma - z
    """
    if isinstance(inp, np.ndarray):
        return squared_hinge_np(inp, targ)
    elif isinstance(inp, torch.Tensor):
        return squared_hinge_torch(inp, targ)
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
