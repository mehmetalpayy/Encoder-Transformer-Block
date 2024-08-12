import torch
import torch.nn as nn

max_sequence_length = 10
d_model = 6

even_i = torch.arange(0, d_model, 2).float()
print(even_i)
even_denominator = torch.pow(10000, even_i / d_model)
print(f"Even Denominator \n{even_denominator}")

odd_i = torch.arange(1, d_model, 2).float()
# we just add 1 to even_i that's why even_denominator and odd_denominator are almost same, and we can choose one of them
odd_denominator = torch.pow(10000, (odd_i - 1) / d_model)

denominator = even_denominator

# position info
position = torch.arange(max_sequence_length, dtype=torch.float).reshape(max_sequence_length, 1)
print("-------Position Information-----------")
print(position)

even_PE = torch.sin(position / denominator)
odd_PE = torch.cos(position / denominator)
print("-------Even Position-----------")
print(even_PE)
print(even_PE.shape)
print("-------ODD Position-----------")
print(odd_PE)
print(odd_PE.shape)

# Stack sin and cos
stacked = torch.stack([even_PE, odd_PE], dim=2)
print("-------STACKED Position-----------")
print(stacked.shape)

PE = torch.flatten(stacked, start_dim=1, end_dim=2)
print("-------FLATTENED-----------")
print(PE)


# Class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE


pe = PositionalEncoding(d_model=6, max_sequence_length=10)
print("-------POSITIONAL ENCODING-----------")
print(pe.forward())
