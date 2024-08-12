import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sequence_length = 4  # Length of the input sentence
batch_size = 1  # batch for parallel processing
input_dim = 512  # vector dimension for every unit and also q,k,v
d_model = 512  # output of the attention unit for every word
x = torch.randn((batch_size, sequence_length, input_dim))
print("-------Size of the Input--------")
print(x.size())

qkv_layer = nn.Linear(input_dim, 3 * d_model)  # multiply 3 to concatenate query, key and value
qkv = qkv_layer(x)  # from (1, 4, 512) to (1, 4, 1536)
print("-------After feed-forward layer--------")
print(qkv.shape)

num_heads = 8  # number of head
head_dim = d_model // num_heads  # dimension for every head
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)
print("-------After 8 head--------")
print(qkv.shape)

# It is important to swap the sequence length with num_heads because the model can evaluate each head separately.
# Also, PyTorch has specific layout requirements for efficient processing.
qkv = qkv.permute(0, 2, 1, 3)  # [batch_size, num_heads, sequence_length, 3*head_dim]
print("-------Permuted shape--------")
print(qkv.shape)

q, k, v = qkv.chunk(3, dim=-1)  # divide the last dimension of qkv into 3 pieces (query, key, value)
print("-------Q, K, V--------")
print(q.shape)
print(k.shape)
print(v.shape)

# Updated self attention part because of shape
d_k = q.size()[-1]
scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (4, 64) x (64, 4)
print("-------Updated self attention--------")
print(scaled.shape)

# Masking
mask = torch.full(scaled.size(), float('-inf'))
mask = torch.triu(mask, diagonal=1)
print("-------Mask for every head--------")
print(mask[0][1])  # mask for input to a single head
print("-------Mask + scaled--------")
print((scaled + mask)[0][0])

scaled += mask
attention = F.softmax(scaled, dim=-1)
print("-------Attention--------")
print(attention[0][0])
print(attention.shape)

# new values
values = torch.matmul(attention, v)
print("-------Values--------")
print(values.shape)


# Functions

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


values, attention = scaled_dot_product(q, k, v, mask=mask)

values = values.reshape(batch_size, sequence_length, num_heads * head_dim)  # concatenate all heads
print("-------Value dimension before feed forward--------")
print(values.shape)
linear_layer = nn.Linear(d_model, d_model)
output = linear_layer(values)
print("-------output dimension--------")
print(output.shape)


# Needed class for encoder_block.py

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out
