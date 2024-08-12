import torch
from torch import nn

inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
B, S, E = inputs.size()
inputs = inputs.reshape(S, B, E)
print("----------INPUTS----------")
print(inputs.size())

parameter_shape = inputs.size()[-2:]
gamma = nn.Parameter(torch.ones(parameter_shape))
beta = nn.Parameter(torch.zeros(parameter_shape))
print("----------GAMMA AND BETA----------")
print(gamma.size())
print(beta.size())

dims = [-(i + 1) for i in range(len(parameter_shape))]
mean = inputs.mean(dim=dims, keepdim=True)
print("----------MEAN SIZE----------")
print(mean.size())

var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
epsilon = 1e-5
std = (var + epsilon).sqrt()
print(std)

y = (inputs - mean) / std
print(y)

out = gamma * y + beta


# Class

class LayerNormalization():
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out
