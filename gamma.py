import torch
GAMMA = 2.2
INV_GAMMA = 1.0/GAMMA
def linear_to_srgb(x):
    return torch.pow(x, INV_GAMMA)

def srgb_to_linear(x):
    return torch.pow(x, GAMMA)
