import numpy as np
import math

# Random Data Generation for query, key and value
# L = length of the input sequence
# d_k = dimension of key and query vectors
# d_v = dimension of value vectors
L, d_k, d_v = 4, 8, 8
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)

# print("Q\n", q)
# print("K\n", k)
# print("V\n", v)

# Self-Attention

x = np.matmul(q, k.T)
print(x)
# We have to minimize the variance to stabilize qxk value that's why we use denominator
print("-----------Before Denominator------------")
print(k.var())
print(q.var())
print(np.matmul(q, k.T).var())
print("-----------After Denominator--------------")
scaled = np.matmul(q, k.T) / math.sqrt(d_k)
print(k.var())
print(q.var())
print(scaled.var())  # it is much more same space with k and q variance anymore.

# Masking

# Masking is optional for encoder because it is much more needed for decoder block in order to predict future tokens.

mask = np.tril(np.ones((L, L)))
print("-----------Mask--------------")
print(mask)
mask[mask == 0] = -np.inf
mask[mask == 1] = 0
print("-----------Adjusted for softmax--------------")
print(mask)
print("-----------Scaled + Mask--------------")
print(scaled + mask)


def softmax(x):
    """Compute softmax values for rows, it is useful for multihead attention."""
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T


attention = softmax(scaled + mask)
print("-----------After Softmax--------------")
print(attention)

# Value after multiplication with self attention
new_v = np.matmul(attention, v)
print("-----------new_v--------------")
print(new_v)


# Functions
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]  # L x d_k -> [-1] == d_k
    scaled = np.matmul(q, k.T) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask
    attention = softmax(scaled)
    out = np.matmul(attention, v)
    return out, attention


values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
print("-----------Last Values--------------")
print(values)
print(attention)
