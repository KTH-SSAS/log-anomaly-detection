"""Code related to Transformer language model."""
import math
from abc import abstractmethod

import torch
from torch import Tensor, nn

from log_analyzer.application import Application
from log_analyzer.config.model_config import TieredTransformerConfig, TransformerConfig
from log_analyzer.model.lstm import LogModel, TieredLogModel
from log_analyzer.model.model_util import initialize_weights


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


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

    def forward(self, sequences, lengths=None, context_vectors=None, mask=None, targets=None):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code

        if not self.bidirectional:
            self.src_mask = self.get_mask(sequences)
        else:
            self.src_mask = None

        word_embeddings = self.word_embedding(sequences) * math.sqrt(self.config.model_dim)
        if context_vectors is not None:
            cat_word_embeddings = torch.Tensor([]).to(sequences.device)
            trans_word_embeddings = word_embeddings.transpose(0, 1).to(sequences.device)
            # Output: trans_word_embeddings: (sequence length x batch x embedded dimension)
            for trans_word_embedding in trans_word_embeddings:
                # trans_word_embedding (batch x embedding)
                trans_word_embedding = torch.cat((trans_word_embedding, context_vectors), dim=-1).unsqueeze(0)
                # Input: trans_word_embedding (batch x embedded dimension), context_vector (batch x context dimension)
                # Output: trans_word_embedding: (1 x batch x embedded dimension + context dimension)
                cat_word_embeddings = torch.cat((cat_word_embeddings, trans_word_embedding), dim=0)
                # Output: cat_word_embeddings: (sequence length x batch x embedded dimension + context dimension)
            trans_cat_word_embeddings = cat_word_embeddings.transpose(0, 1)
            # Output: trans_cat_word_embeddings: (batch x sequence length x embedded dimension + context dimension)
            word_embeddings = self.reduce_dimension(trans_cat_word_embeddings)
            # Output: word_embeddings: (batch x sequence length x embedded dimension)
            word_embeddings = word_embeddings * math.sqrt(self.config.model_dim)
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
        self.layers = config.layers
        self.attention_heads = config.attention_heads
        self.feedforward_dim = config.feedforward_dim
        self.vocab_size = config.vocab_size
        self.name = "Tiered_Transformer"
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout)
        self.word_embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.transformer_model = nn.Transformer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            num_encoder_layers=config.layers,
            num_decoder_layers=config.context_config.layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.shift_window = config.shift_window + 1
        # To use transformer structure, the input lenghth for transformer encoder should be 1+.

        # User model state
        self.context_model_dim = config.context_config.model_dim
        self.context_input_dimension = config.input_dim
        self.n_users = config.number_of_users
        self.reduce_dim = self.reduce_dimension = nn.Linear(config.context_dim, self.model_dim)
        self.saved_context_histories = torch.zeros([self.n_users, self.shift_window, config.context_dim])
        self.saved_context_history_lengths = torch.ones([self.n_users], dtype=torch.int16)

        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def gen_mask(self, seq_len, device=None, mask=None, has_mask=True):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code
        if has_mask:
            mask = _generate_square_subsequent_mask(seq_len).to(device)
        else:
            mask = None
        return mask

    def gen_pad_mask(self, ctx_history, history_length, device=None, has_mask=True):
        if has_mask:
            mask = torch.ones([ctx_history.shape[0], ctx_history.shape[1]]) == 1
            for p, i in zip(history_length, range(mask.shape[0])):
                mask[i, :p] = False
        else:
            mask = None
        return mask.to(device)

    def forward(self, sequences: Tensor, lengths=None, mask=None, targets=None):
        users, src = sequences
        # Convert users list to python list
        users = [user.item() for user in users]

        # Get the state for the users in the batch
        context_history, history_length = self.get_batch_data(users, src.device)

        # Get the number of steps in the batch
        self.num_steps = src.shape[0]

        shape = src.shape + (self.config.vocab_size,)
        token_output = torch.zeros(shape, dtype=torch.float).to(src.device)

        # token_output = token_output.unsqueeze(3).repeat(1, 1, 1, self.config.vocab_size)

        for idx, batch in enumerate(src):
            tgt_mask = self.gen_mask(batch.shape[-1], device=src.device)
            src_compressed = self.reduce_dim(context_history)

            src_pad_mask = self.gen_pad_mask(src_compressed, history_length, device=src.device)
            embedding_tgt_input = self.word_embedding(batch) * math.sqrt(self.model_dim)

            src_input = self.pos_encoder(src_compressed * math.sqrt(self.model_dim))[
                :, list(range(src_compressed.shape[1]))[::-1], :
            ]
            # invert the order of the context
            # e.g., for 2 users and 5 shift window, the context history would be
            # 0 0 0 0 i ==> i 0 0 0 0
            # 0 0 0 i i ==> i i 0 0 0
            # So we can apply padding mask to history input as
            # F T T T T
            # F F T T T
            # where T stands for True and padding, and F stands for False and no padding.
            tgt_input = self.pos_encoder(embedding_tgt_input)

            tf_hidden = self.transformer_model(
                src=src_input, src_key_padding_mask=src_pad_mask, tgt=tgt_input, tgt_mask=tgt_mask
            )
            context_input = torch.unsqueeze(
                torch.cat([tf_hidden[:, -1, :], torch.mean(tf_hidden, dim=1)], dim=-1), dim=1
            )
            new_context_history = torch.cat([context_history, context_input], dim=1)
            history_length = torch.min(
                history_length + 1, torch.ones(history_length.shape, dtype=torch.int16) * self.shift_window
            )
            if self.shift_window == 1:
                context_history = context_history
            else:
                context_history = new_context_history[:, -max(history_length) :, :]
            logits = tf_hidden @ self.word_embedding.weight.t()
            token_output[idx][: logits.shape[0], : logits.shape[1], : logits.shape[2]] = logits
        # Update context state
        self.update_state(users, context_history, history_length)

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, _ = self.compute_loss(token_output, targets)
        return token_output, loss

    def get_batch_data(self, users, device):
        """Given a list of users, fetch the relevant history and model data for
        each user."""
        history = self.saved_context_histories[torch.tensor(users)]
        history_lengths = self.saved_context_history_lengths[torch.tensor(users)]
        # Crop the length of history returned to max history_length amongst users in this batch
        max_length = torch.max(history_lengths)
        return history[:, -max_length:, :].to(device), history_lengths

    def update_state(self, users, context_history, history_length):
        """Given one batch of history/model data output by the model, update
        the stored state for future use."""
        context_history = context_history.detach().cpu()
        self.saved_context_history_lengths[torch.tensor(users)] = history_length
        max_length = torch.max(self.saved_context_history_lengths[torch.tensor(users)])
        self.saved_context_histories[torch.tensor(users), -(context_history.shape[1]) :, :] = context_history
