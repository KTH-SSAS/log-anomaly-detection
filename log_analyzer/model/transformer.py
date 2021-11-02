"""Code related to Transformer language model"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from log_analyzer.config.model_config import TransformerConfig
from log_analyzer.model.lstm import LogModel
from log_analyzer.model.model_util import initialize_weights


# Positional Encoding class taken from PyTorch word_language_model example code:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model=None, dropout=0.1, max_len=None):
        super(NoPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x)

# Original TransformerModel code taken from PyTorch word_language_model example code:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py


class Transformer(LogModel):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, config: TransformerConfig):
        self.name = "Transformer"
        super().__init__(config)

        self.config = config

        self.dropout = config.dropout
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(
            config.model_dim, dropout=self.dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            config.model_dim, config.attention_heads, config.feedforward_dim, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, config.layers)
        self.word_embedding = nn.Embedding(config.vocab_size, config.model_dim)

        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor, lengths=None,
                mask=None, has_mask=True):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code
        # TODO: compatibility with character level encoding
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.shape[-1] != src.shape[-1]:
                mask = self._generate_square_subsequent_mask(
                    src.shape[-1]).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        word_embeddings = self.word_embedding(
            src) * math.sqrt(self.config.model_dim)
        tf_input = self.pos_encoder(word_embeddings)
        tf_hidden = self.transformer_encoder(tf_input, self.src_mask)
        # word embedding encoder and decoder share weights
        logits = tf_hidden @ self.word_embedding.weight.t()
        # Trainer expects model to return a tuple of results (for the LSTMs this would be (lstm_out, final_hidden_state))
        # So we have to return a tuple here too (all but the first value of the
        # tuple are discarded)
        return logits, []
