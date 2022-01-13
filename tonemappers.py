import torch
from torch import nn as nn

from gamma import linear_to_srgb

class UC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = 0.15
        self.B = 0.50
        self.C = 0.10
        self.D = 0.20
        self.E = 0.02
        self.F = 0.30
        self.W = 11.2
        self.white = ((self.W * (self.A * self.W + self.C * self.B) + self.D * self.E) /
                      (self.W * (self.A * self.W + self.B) + self.D * self.F)) - self.E / self.F

    def forward(self, x):
        x = ((x * (self.A * x + self.C * self.B) + self.D * self.E) /
             (x * (self.A * x + self.B) + self.D * self.F)) - self.E / self.F
        x = x / self.white
        return linear_to_srgb(x)


class ACES(nn.Module):
    def __init__(self):
        super().__init__()
        a_in = torch.tensor([[0.59719, 0.07600, 0.02840],
                             [0.35458, 0.90834, 0.13383],
                             [0.04823, 0.01566, 0.83777],
                             ])
        a_out = torch.tensor([[1.60475, -0.10208, -0.00327],
                              [-0.53108, 1.10813, -0.07276],
                              [-0.07367, -0.00605, 1.07602],
                              ])
        self.aces_in = nn.Parameter(a_in, requires_grad=False)
        self.aces_out = nn.Parameter(a_out, requires_grad=False)

    def forward(self, x):
        x = x @ self.aces_in
        a = x * (x + 0.0245786) - 0.000090537
        b = x * (0.983729 * x + 0.4329510) + 0.238081
        x = a / b
        x = x @ self.aces_out
        return linear_to_srgb(torch.clamp(x,0.0,1.0))


class ACES_Approx(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 2.51
        self.b = 0.03
        self.c = 2.43
        self.d = 0.59
        self.e = 0.14

    def forward(self, x):
        x = x * 0.6
        x = torch.clamp((x * (self.a * x + self.b)) / (x * (self.c * x + self.d) + self.e), 0.0, 1.0)
        return linear_to_srgb(x)


class Cineon(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x - 0.004, 0.0, 99999.0)
        x = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
        # linear to sRGB conversion embedded in shader
        return x


class JodieLuma(nn.Module):
    def __init__(self):
        super().__init__()
        self.dotp = nn.Parameter(torch.tensor((0.2126, 0.7152, 0.0722)))

    def forward(self, x):
        luma = (x * self.dotp).sum(dim=-1,keepdim=True)
        tm_luma = luma / (luma + 1.0)
        x = x * (tm_luma / luma)
        return linear_to_srgb(x)


class ReinhardExtended(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_w = 1.0

    def forward(self, x):
        num = x * (self.max_w + (x / (self.max_w ** 2)))
        x = num / (1 + x)
        return linear_to_srgb(x)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return linear_to_srgb(torch.clamp(x, 0.0, 1.0))
