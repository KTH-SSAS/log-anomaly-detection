"""Code related to different forms of attention"""
import torch.nn as nn
import torch


def generate_mask(seq_len, hidden_dim):
    """Generate mask for unidirectional attention"""
    mask = torch.triu(torch.ones(hidden_dim, seq_len, seq_len))
    return mask.unsqueeze(0)  # Unsqueeze to allow broadcasting over batch


def generate_softmax_mask(seq_len):
    """
    Generate mask that prevents the zeros in a lower triangular matrix from being counted when applying the softmax function.
    Creates a matrix where the lower diagonal are zeros, and the upper diagonal are -inf (to make e^x 0).
    The mask is applied by adding it to the input matrix.
    """
    softmax_mask = ((torch.tril(torch.ones(seq_len, seq_len),
                    diagonal=0) != 1) * -float('inf')).nan_to_num(nan=0)
    # To allow mask to be broadcasted across q
    softmax_mask = softmax_mask.unsqueeze(1)
    return softmax_mask


class SelfAttention(nn.Module):
    """Self-attention (mostly as described in Brown paper)"""

    def __init__(self, hidden_dim, attention_dim, attention_type, max_len=200):
        super().__init__()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.w_a = nn.Linear(hidden_dim, attention_dim, bias=False)
        # TODO add the other types
        self.attention_type = 'fixed'
        if attention_type == 'fixed':
            self.query = nn.Linear(
                attention_dim, 1, bias=False)  # Fixed attention
        elif attention_type == 'syntax':
            raise NotImplementedError(
                'Syntax attention is not yet functioning.')
            self.query = nn.Linear(attention_dim, max_len, bias=False)

    def forward(self, x):
        """Calculate attention weights for input sequence"""
        seq_len = x.shape[1]
        mask = generate_mask(seq_len, hidden_dim=x.shape[-1])

        v = x.unsqueeze(1)
        v = v.repeat(1, seq_len, 1, 1)

        v_masked = torch.mul(mask, v.transpose(1, -1))

        key = self.tanh(self.w_a(v_masked.transpose(1, -1)))

        temp = torch.matmul(self.query.weight, key.transpose(-2, -1))

        temp = temp + generate_softmax_mask(seq_len)

        d = self.softmax(temp)

        a = torch.matmul(d, v_masked.transpose(1, -1))

        return a, d
