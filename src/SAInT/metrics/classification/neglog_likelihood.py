import torch
import numpy as np

# https://arxiv.org/pdf/2301.05579

def neglog_likelihood_np(inp, targ):
    """Negative Log-Likelihood Loss: NumPy implementation"""
    assert inp.shape == targ.shape, "Shapes of input probabilities and targets must match"
    epsilon = 1e-15
    # Ensure probabilities are within valid range to prevent log(0)
    inp = np.clip(inp, epsilon, 1 - epsilon)
    # Compute Negative Log-Likelihood
    nll_loss = -np.sum(targ * np.log(inp), axis=1)
    return np.mean(nll_loss)

def neglog_likelihood_torch(inp, targ):
    """Negative Log-Likelihood Loss: PyTorch implementation"""
    assert inp.shape == targ.shape, "Shapes of input probabilities and targets must match"
    epsilon = 1e-15
    # Ensure probabilities are within valid range to prevent log(0)
    inp = torch.clamp(inp, min=epsilon, max=1-epsilon)
    # Compute Negative Log-Likelihood
    nll_loss = -torch.sum(targ * torch.log(inp), dim=1)
    return torch.mean(nll_loss)

def neglog_likelihood(inp, targ):
    """Negative Log-Likelihood Loss"""
    if isinstance(inp, np.ndarray) and isinstance(targ, np.ndarray):
        return neglog_likelihood_np(inp, targ)
    elif isinstance(inp, torch.Tensor) and isinstance(targ, torch.Tensor):
        return neglog_likelihood_torch(inp, targ)
    else:
        raise TypeError("Inputs should be either NumPy arrays or PyTorch tensors.")
