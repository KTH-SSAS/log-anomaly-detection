"""Code related to different forms of attention"""
from torch.functional import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple

def generate_mask(seq_len, hidden_dim):
    """Generate mask for unidirectional attention"""
    mask = torch.triu(torch.ones(hidden_dim, seq_len, seq_len))
    return mask.unsqueeze(0)  # Unsqueeze to allow broadcasting over batch


def generate_softmax_mask(seq_len):
    """
    Generate mask that prevents the zeros in a lower triangular matrix from being counted when applying the softmax function.
    Creates a matrix where the lower diagonal are zeros, and the upper diagonal are -inf (to make e^x 0).
    The mask is applied by adding it to the input matrix.
    Example for a 3x3 matrix:
        0 -inf -inf
        0   0  -inf
        0   0    0 

    """
    softmax_mask = ((torch.tril(torch.ones(seq_len, seq_len),
                    diagonal=0) != 1) * -float('inf')).nan_to_num(nan=0)
    return softmax_mask

class SelfAttention(nn.Module):
    """Self-attention (mostly as described in Brown paper)"""

    def __init__(self, hidden_dim, attention_dim, attention_type, seq_len=None):
        super().__init__()
        self.w_a = nn.Parameter(torch.Tensor(hidden_dim, attention_dim))
        # TODO add the other types
        self.attention_type = attention_type
        if attention_type == 'fixed':
            self.query = nn.Parameter(torch.Tensor(attention_dim))  # Fixed attention
        elif attention_type == 'syntax':
            if seq_len is None:
                raise RuntimeError('For syntax attention a sequence length has to bet set.')
            self.query = nn.Parameter(torch.Tensor(seq_len, attention_dim))
        elif attention_type == 'semantic':
            self.query = nn.Parameter(torch.Tensor(hidden_dim, attention_dim))

    def forward(self, x: torch.Tensor):
        """Calculate attention weights for input sequence"""
        seq_len = x.shape[1]
        if self.attention_type == 'fixed': # For fixed attention, the hidden states are replicated and masked in order to get an attention matrix of size LxL
            mask = generate_mask(seq_len, hidden_dim=x.shape[-1])
            x_repeat = x.unsqueeze(1).repeat(1, seq_len, 1, 1)
            x_masked = torch.mul(mask, x_repeat.transpose(1, -1))
            values = x_masked.transpose(1, -1)
        else:
            values = x
        
        key = torch.tanh(torch.matmul(values, self.w_a))

        if self.attention_type == 'fixed' or self.attention_type == 'syntax':
            q = self.query
        else:
            q = torch.matmul(values, self.query)

        temp = torch.matmul(q, key.transpose(-2, -1))

        softmax_mask = generate_softmax_mask(seq_len)

        temp = temp + softmax_mask

        d = F.softmax(temp, dim=-1)

        a = torch.matmul(d, x)

        return a, d


def save_attention_graph(attention_matrix: Tensor):
    plt.matshow(attention_matrix.detach().numpy().mean(axis=0)) # Average over batch
    plt.show()