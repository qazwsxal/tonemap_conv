import torch
GAMMA = 2.2
INV_GAMMA = 1.0/GAMMA
# We use abs calls so things don't break around 0
def linear_to_srgb(x):
    return torch.pow(x.abs(), INV_GAMMA)

def srgb_to_linear(x):
    return torch.pow(x.abs(), GAMMA)
