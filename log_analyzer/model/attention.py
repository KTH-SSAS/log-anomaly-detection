"""Code related to different forms of attention."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.functional import Tensor

from log_analyzer.application import Application


def generate_mask(seq_len, hidden_dim, use_cuda=False):
    """Generate mask for unidirectional attention."""
    mask = torch.triu(torch.ones(hidden_dim, seq_len, seq_len))
    mask = mask.cuda() if use_cuda else mask
    return mask.unsqueeze(0)  # Unsqueeze to allow broadcasting over batch


def generate_softmax_mask(seq_len, use_cuda=False):
    """Generate mask that prevents the zeros in a lower triangular matrix from
    being counted when applying the softmax function.

    Creates a matrix where the lower diagonal are zeros, and the upper
    diagonal are -inf (to make e^x 0). The mask is applied by adding it
    to the input matrix. Example for a 3x3 matrix:     0 -inf -inf     0
    0  -inf     0   0    0
    """
    softmax_mask: torch.Tensor = (
        (torch.tril(torch.ones(seq_len, seq_len), diagonal=0) != 1) * -float("inf")
    ).nan_to_num(nan=0)
    softmax_mask = softmax_mask.cuda() if use_cuda else softmax_mask
    return softmax_mask

FIXED = 0
SEMANTIC = 1
SYNTAX = 2


attention_names = {
    "fixed": FIXED,
    "semantic": SEMANTIC,
    "syntax": SYNTAX
}


def get_query_dim(attention_type, seq_len, hidden_dim, attention_dim):

    if attention_type == FIXED:
        # Shared one-dimension vector
        return (attention_dim,)
    if attention_type == SYNTAX:
        if seq_len is None:
            raise RuntimeError("For syntax attention a sequence length has to bet set.")
        # One query vector per position in sequence
        return (seq_len, attention_dim)
    if attention_type == SEMANTIC:
        # One query vector per hidden unit
        return (hidden_dim, attention_dim)

    raise RuntimeError("Unknown attention type")


class SelfAttention(nn.Module):
    """Self-attention (mostly as described in Brown paper)"""

    def __init__(self, hidden_dim, attention_dim, attention_type, seq_len=None):
        super().__init__()
        self.using_cuda = Application.instance().using_cuda
        self.w_a = nn.Parameter(torch.Tensor(hidden_dim, attention_dim))
        torch.nn.init.xavier_normal_(self.w_a)
        self.attention_type = attention_names[attention_type]
        # Depending on the type of attention, the query vector has different
        # dimensions

        query_dim = get_query_dim(self.attention_type, seq_len, hidden_dim, attention_dim)
        self.query = nn.Parameter(torch.empty(*query_dim))
        if self.attention_type == FIXED:
            torch.nn.init.normal_(self.query)
        else:
            torch.nn.init.xavier_normal_(self.query)

        if seq_len is not None:  # If the input length is fixed, we can cache the masks
            self.input_mask = generate_mask(seq_len, hidden_dim, self.using_cuda)
            self.softmax_mask = generate_softmax_mask(seq_len, self.using_cuda)
        else:
            self.input_mask = None
            self.softmax_mask = None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        """Calculate attention weights for input sequence.

        For fixed attention, the hidden states are replicated and masked
        in order to get an attention matrix of size LxL
        """
        seq_len = x.shape[1]
        if self.attention_type == FIXED:

            mask = (
                generate_mask(seq_len, hidden_dim=x.shape[-1], use_cuda=self.using_cuda)
                if self.input_mask is None
                else self.input_mask
            )
            x_repeat = x.unsqueeze(1).repeat(1, seq_len, 1, 1)
            x_masked = torch.mul(mask, x_repeat.transpose(1, -1))
            values = x_masked.transpose(1, -1)
            key = torch.tanh(torch.einsum("bijd,da->bija", values, self.w_a))
        else:
            values = x
            # b: batch size, j: seq len, a: attention dim
            key = torch.tanh(torch.einsum("bjd,da->bja", values, self.w_a))

        # b: batch size, i,j: seq_len, a: attention dim
        if self.attention_type == SYNTAX:
            temp = torch.einsum("ia,bja->bij", self.query, key)
        elif self.attention_type == FIXED:
            temp = torch.einsum("a,bija->bij", self.query, key)
        else:
            temp = torch.einsum("bid,da,bja->bij", values, self.query, key)

        softmax_mask = (
            generate_softmax_mask(seq_len, use_cuda=self.using_cuda) if self.softmax_mask is None else self.softmax_mask
        )  # Use cached mask for fixed length seqs

        temp = temp + softmax_mask
        d = F.softmax(temp, dim=-1)

        # Padding mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len] != 0
            # Create a 2D matrix
            attention_mask = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
            d = torch.mul(attention_mask, d)

        a = torch.einsum("bij,bjd->bjd", d, x)

        return a, d


def save_attention_graph(attention_matrix: Tensor):
    plt.matshow(attention_matrix.detach().numpy().mean(axis=0))  # Average over batch
    plt.show()
