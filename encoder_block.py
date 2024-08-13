import torch
import math
from torch import nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    """
    We calculate the similarity between the query and key vector, and we divide it to square root of k dimension
    to reduce the variance (around 0-1). We apply softmax function to every row and getting new values by
    calculating the multiplication of attention with v.

    Params:
        q, k, v: At the beginning all the tensors that we use as query, key and value will be same
                the dimension of tensors batch size x number of attention heads x sequence length x
                length of the query vector of each head.
        mask: In encoder block no required to use mask but for decoder block it is important that's why we leave it
        as optional

    Returns:
        values and attention
    """
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    """
    This class implements the Multi-Head Attention mechanism.

    Params:
        input_dim: The vector dimension of each word that goes into the attention unit.
        d_model: The output dimension of the attention unit for each word
        num_heads: The number of attention heads.

    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Params:
            x: tensor from positional encoding of size batch_size, sequence_length, input_dim
            mask: optinal mask
                In encoder block no required to use mask but for decoder block it is important that's why we leave it
                as optional

        Returns:
            out: concatenated value tensor
        """
        batch_size, max_sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    """
    This class implements Layer Normalization along the feature dimension of the word embeddings.
    It ensures that the values remain consistent

    Params:
        parameters_shape: The shape of the parameters (gamma and beta) used for normalization.
        eps: A small constant added to the denominator to prevent division by zero.
    """
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        """
        Params:
            inputs: tensor of word embeddings

        Returns:
            out: Normalized output
        """
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


class ResidualConnection(nn.Module):
    """
    This class implements a residual connection with layer normalization, commonly used in Transformer models.
    It adds a residual connection around a sublayer and normalizes the result.

    Params:
        d_model: The dimensionality of the input embeddings.
        dropout: The probability of dropping out units during training, used for regularization.
    """
    def __init__(self, d_model, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNormalization(parameters_shape=[d_model])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer_output):
        """
        Params:
            x: The input tensor of shape `(batch_size, sequence_length, d_model)`.
            sublayer_output: The output tensor from the sublayer, of shape `(batch_size, sequence_length, d_model)`.

        Returns:
            normalized: The tensor after applying the residual connection, dropout, and layer normalization.
            The final output tensor has the same shape as the input tensor `(batch_size, sequence_length, d_model)`.
        """
        d_out = self.dropout(sublayer_output)
        residual = x + d_out
        normalized = self.norm(residual)
        return normalized


class PositionwiseFeedForward(nn.Module):
    """
    This class implements a position-wise feed-forward network, which is a component commonly used in Transformer models.
    It consists of two linear transformations with a ReLU activation and dropout for regularization.

    Params:
        d_model: The dimensionality of the input and output embeddings.
        hidden: The number of units in the hidden layer.
        dropout: The probability of dropping out units during training, used for regularization. Default is 0.1.
    """

    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Params:
            x: Input tensor of shape `(batch_size, sequence_length, d_model)`.

        Returns:
            output: Output tensor after applying two linear transformations, a ReLU activation, and dropout.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.linear2(x)
        return output


class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.residual1 = ResidualConnection(d_model=d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=d_ff, dropout=dropout)
        self.residual2 = ResidualConnection(d_model=d_model, dropout=dropout)

    def forward(self, x, mask=None):
        # Multi-head attention + residual connection
        multiheadattention = self.attention(x, mask=mask)
        x = self.residual1(x, multiheadattention)

        # Feed-forward network + residual connection
        feedforward = self.ffn(x)
        x = self.residual2(x, feedforward)
        return x
