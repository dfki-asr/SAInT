import torch
import numpy as np

def logcosh(inp, targ):
    """Log cosh error
    the original log cosh error is defined as:
        mean(log(cosh(inp-targ)))
    it can be infinity with high tensor differences.

    The stable log cosh error version:
        mean( (inp-targ) + log(cosh(inp-targ)) - log(2.0))
    """
    assert inp.shape == targ.shape, "Shapes of input and target must match"
    if isinstance(inp, np.ndarray):
        diff = inp - targ
        return np.mean(diff + np.log(np.cosh(diff)) - np.log(2.0))
    elif isinstance(inp, torch.Tensor):
        diff = inp - targ
        return torch.mean(diff + torch.nn.functional.softplus(-2 * diff) - torch.log(torch.tensor(2.0)))
    else:
        raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
