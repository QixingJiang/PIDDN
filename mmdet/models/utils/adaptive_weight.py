import torch
import torch.nn as nn


class AdaptiveWeight(nn.Module):

    def __init__(self, weight=1.0):
        super(AdaptiveWeight, self).__init__()
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float))

    def forward(self, x):
        return x * self.weight

    def __repr__(self):
        return f'the weight is {self.weight}'





