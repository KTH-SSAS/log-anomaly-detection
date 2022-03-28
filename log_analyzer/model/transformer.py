"""Code related to Transformer language model."""
import math
from abc import abstractmethod

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from log_analyzer.application import Application
from log_analyzer.config.model_config import TieredTransformerConfig, TransformerConfig
from log_analyzer.model.lstm import LogModel, TieredLogModel
from log_analyzer.model.model_util import initialize_weights


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def _generate_padding_mask(ctx_history, history_length):
    mask = torch.ones(ctx_history.shape[:-1]) == 1
    for p, i in zip(history_length, range(mask.shape[0])):
        mask[i, :p] = False
    return mask.to(ctx_history.device)


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
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.using_cuda = Application.instance().using_cuda
        self.register_buffer("pe", pe)
        self.pe: Tensor

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
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class NoPositionalEncoding(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x)


# Original TransformerModel code taken from PyTorch word_language_model example code:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py


class TransformerLanguageModel(LogModel):
    """Container module with an encoder, a recurrent or transformer module, and
    a decoder."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config: TransformerConfig = config
        self.src_mask = None

        self.dropout = config.dropout
        self.model_dim = config.model_dim
        self.layers = config.layers
        self.attention_heads = config.attention_heads
        self.feedforward_dim = config.feedforward_dim
        self.vocab_size = config.vocab_size

        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            self.model_dim, self.attention_heads, self.feedforward_dim, dropout=self.dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.layers)

    def get_mask(self, src: torch.Tensor):
        # batch size, sequence length, embedded dimension
        seq_len = src.shape[-1]
        device = src.device
        if self.src_mask is None or self.src_mask.shape[-1] != seq_len:
            mask = _generate_square_subsequent_mask(seq_len).to(device)
            self.src_mask = mask
        return self.src_mask

    @abstractmethod
    def forward(self, sequences, lengths: Tensor = None, context_vectors=None, mask=None, targets=None):
        ...


class Transformer(TransformerLanguageModel):
    """Container module with an encoder, a recurrent or transformer module, and
    a decoder."""

    def __init__(self, config: TransformerConfig, bidirectional):
        self.name = "Transformer"
        super().__init__(config)
        self.bidirectional = bidirectional
        self.word_embedding = nn.Embedding(self.vocab_size, self.model_dim)
        if isinstance(config, TieredTransformerConfig):
            self.reduce_dimension = nn.Linear(config.input_dim, self.model_dim)
            if Application.instance().using_cuda:
                self.reduce_dimension = self.reduce_dimension.cuda()
        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def forward(self, sequences, lengths=None, mask=None, targets=None):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code

        if not self.bidirectional:
            self.src_mask = self.get_mask(sequences)
        else:
            self.src_mask = None

        word_embeddings = self.word_embedding(sequences) * math.sqrt(self.config.model_dim)
        tf_input = self.pos_encoder(word_embeddings)
        if mask is None:
            pad_mask = None
        else:
            pad_mask = mask == 0
        tf_hidden = self.transformer_encoder(tf_input, self.src_mask, src_key_padding_mask=pad_mask)
        # word embedding encoder and decoder share weights
        # @ is shorthand for matrix multiplication
        logits = tf_hidden @ self.word_embedding.weight.t()

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, _ = self.compute_loss(logits, targets)

        if self.tiered:
            return logits, tf_hidden, loss
        return logits, loss


class TieredTransformer(TieredLogModel):

    num_steps: int

    def __init__(self, config: TieredTransformerConfig, bidirectional):
        super().__init__(config)
        self.bidirectional = bidirectional
        self.dropout = config.dropout
        self.model_dim = config.model_dim
        self.ctx_dim = config.input_dim
        self.name = "Tiered_Transformer"
        self.src_pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout)
        self.tgt_pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout)
        self.word_embedding = nn.Embedding(config.vocab_size, self.model_dim)
        self.transformer_model = nn.Transformer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            num_encoder_layers=config.layers,
            num_decoder_layers=config.layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.shift_window = config.shift_window
        # To use transformer structure, the input lenghth for transformer encoder should be 1+.

        # User model state
        self.n_users = config.number_of_users
        self.reduce_dim = nn.Linear(self.ctx_dim, self.model_dim)
        self.src_mask = None
        self.tgt_mask = None
        self.ctx_histories = []
        for u in range(self.n_users):
            self.ctx_histories.append(torch.zeros([1, self.ctx_dim]))
        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def get_src_mask(self, src: torch.Tensor):
        # batch size, sequence length, embedded dimension
        seq_len = src.shape[1]
        device = src.device
        mask = _generate_square_subsequent_mask(seq_len).to(device)
        return mask

    def get_tgt_mask(self, tgt: torch.Tensor):
        # batch size, sequence length, embedded dimension
        seq_len = tgt.shape[-1]
        device = tgt.device
        if self.tgt_mask is None or self.tgt_mask.shape[-1] != seq_len:
            self.tgt_mask = _generate_square_subsequent_mask(seq_len).to(device)
        return self.tgt_mask

    def forward(self, sequences, lengths=None, mask=None, targets=None):
        users, tgt = sequences
        # Convert users list to python list
        users = [user.item() for user in users]

        # Get the state for the users in the batch
        ctx_histories = self.get_ctx_data(users)

        # Get the number of steps in the batch
        self.num_steps = tgt.shape[0]

        shape = tgt.shape + (self.config.vocab_size,)
        token_output = torch.zeros(shape, dtype=torch.float).to(tgt.device)

        # token_output = token_output.unsqueeze(3).repeat(1, 1, 1, self.config.vocab_size)
        # src = context info, tgt = log line info
        for idx, batch in enumerate(tgt):
            tgt_input = self.tgt_pos_encoder(self.word_embedding(batch) * math.sqrt(self.model_dim))

            src_padded = pad_sequence(ctx_histories, batch_first=True).to(tgt.device)
            src_compressed = self.reduce_dim(src_padded)
            src_input = self.src_pos_encoder(src_compressed * math.sqrt(self.model_dim))
            src_pad_mask = torch.all(src_padded == 0, dim=-1)
            src_pad_mask[:, 0] = False
            src_mask = self.get_src_mask(src_padded)
            tgt_mask = self.get_tgt_mask(batch)

            # e.g.,
            # i 0 0 0 0
            # i i 0 0 0
            # So we can apply padding mask to history input as
            # F T T T T
            # F F T T T
            # where T stands for True and padding, and F stands for False and no padding.

            tf_hidden = self.transformer_model(
                src=src_input, src_mask=src_mask, src_key_padding_mask=src_pad_mask, tgt=tgt_input, tgt_mask=tgt_mask
            )
            ctx_inputs = torch.cat([torch.mean(tf_hidden, dim=1), tf_hidden[:, -1, :]], dim=-1)
            for i, ctx in enumerate(ctx_inputs):
                ctx_histories[i] = torch.cat([ctx_histories[i], ctx.reshape(1, -1)], dim=0)[-self.shift_window :, :]
            if self.shift_window == 0:
                ctx_histories = torch.zeros([tgt.shape[0], 1, self.ctx_dim])
            logits = tf_hidden @ self.word_embedding.weight.t()
            token_output[idx][: logits.shape[0], : logits.shape[1], : logits.shape[2]] = logits
        # Update context state
        self.update_ctx_data(users, ctx_histories)

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, _ = self.compute_loss(token_output, targets)
        return token_output, loss

    def get_ctx_data(self, users):
        """Given a list of users, fetch the relevant history and model data for
        each user."""
        history = []
        for u in users:
            history.append(self.ctx_histories[u])
        return history

    def update_ctx_data(self, users, ctx_histories):
        for u, ctx_history in zip(users, ctx_histories):
            self.ctx_histories[u] = ctx_history
