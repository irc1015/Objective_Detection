#version ==> 0.0.1-2025.6
#https://arxiv.org/pdf/2407.08567
#https://github.com/kostas1515/AGLU

import torch
import torch.nn as nn

class AGLU(nn.Module):
    '''
    Attributes:
        act (nn.Softplus): Softplus activation function with negative beta.
        lambd (nn.Parameter): Learnable lambda parameter initialized with uniform distribution.
        kappa (nn.Parameter): Learnable kappa parameter initialized with uniform distribution.
    '''
    def __init__(self, devive=None, dtype=None) -> None:
        super().__init__()
        self.act = nn.Softplus(beta=1.0)
        self.lambd = nn.parameter(nn.init.uniform_(torch.empty(1, device=devive, dtype=dtype)))
        self.kappa = nn.parameter(nn.init.uniform_(torch.empty(1, device=devive, dtype=dtype)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp(1 / lam) * self.act((self.kappa * x) - torch.log(lam))