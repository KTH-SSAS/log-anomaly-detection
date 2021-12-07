"""Code related to Transformer language model."""
import math

import torch
from torch import Tensor, nn

from log_analyzer.application import Application
from log_analyzer.config.model_config import TieredTransformerConfig, TransformerConfig
from log_analyzer.model.lstm import LogModel
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
        self.cuda = Application.instance().using_cuda
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

    def forward(self, src: torch.Tensor, lengths=None, mask=None, has_mask=True):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code
        if isinstance(self, ContextTransformer):
            seq_len = src.shape[1]
        else:
            seq_len = src.shape[-1]
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.shape[-1] != seq_len:
                mask = _generate_square_subsequent_mask(seq_len).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        return self.src_mask


class Transformer(TransformerLanguageModel):
    """Container module with an encoder, a recurrent or transformer module, and
    a decoder."""

    def __init__(self, config: TransformerConfig):
        self.name = "Transformer"
        super().__init__(config)
        self.bidirectional = False  # TODO: Change this when we make a bidirectional model.
        self.word_embedding = nn.Embedding(self.vocab_size, self.model_dim)
        if isinstance(config, TieredTransformerConfig):
            self.reduce_dimension = nn.Linear(config.input_dim, self.model_dim)
            if Application.instance().using_cuda:
                self.reduce_dimension = self.reduce_dimension.cuda()
        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def forward(self, src, ctx_vector=None, lengths=None, mask=None, has_mask=True):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code
        # TODO: compatibility with character level encoding

        self.src_mask = super().forward(src, has_mask)
        word_embeddings = self.word_embedding(src) * math.sqrt(self.config.model_dim)
        if ctx_vector is not None:
            cat_word_embeddings = torch.Tensor([]).to(src.device)
            trans_word_embeddings = word_embeddings.transpose(0, 1).to(src.device)
            # Output: trans_word_embeddings: (sequence length x batch x embedded dimension)
            for trans_word_embedding in trans_word_embeddings:
                # trans_word_embedding (batch x embedding)
                trans_word_embedding = torch.unsqueeze(torch.cat((trans_word_embedding, ctx_vector), dim=1), dim=0)
                # Input: trans_word_embedding (batch x embedded dimension), ctx_vector (batch x context dimension)
                # Output: trans_word_embedding: (1 x batch x embedded dimension + context dimension)
                cat_word_embeddings = torch.cat((cat_word_embeddings, trans_word_embedding), dim=0)
                # Output: cat_word_embeddings: (sequence length x batch x embedded dimension + context dimension)
            trans_cat_word_embeddings = cat_word_embeddings.transpose(0, 1)
            # Output: trans_cat_word_embeddings: (batch x sequence length x embedded dimension + context dimension)
            word_embeddings = self.reduce_dimension(trans_cat_word_embeddings)
            # Output: word_embeddings: (batch x sequence length x embedded dimension)
        tf_input = self.pos_encoder(word_embeddings)
        tf_hidden = self.transformer_encoder(tf_input, self.src_mask)
        # word embedding encoder and decoder share weights
        logits = tf_hidden @ self.word_embedding.weight.t()
        # Trainer expects model to return a tuple of results (for the LSTMs this would be (lstm_out, final_hidden_state))
        # So we have to return a tuple here too (all but the first value of the tuple are discarded)
        return logits, tf_hidden  # 2nd output (tf hidden) for context transformer.


class ContextTransformer(TransformerLanguageModel):
    """Container module with an encoder, a recurrent or transformer module, and
    a decoder."""

    def __init__(self, config: TieredTransformerConfig):
        self.name = "Context_Transformer"
        super().__init__(config.context_config)
        self.reduce_dimension = nn.Linear(2 * config.model_dim, config.context_config.model_dim)
        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def forward(self, ctx_history, lengths=None, mask=None, has_mask=True):

        self.src_mask = super().forward(ctx_history, has_mask)
        ctx_input = self.reduce_dimension(ctx_history)  # ctx_input (batch size, sequence length, 2 * model dimension)
        ctx_embeddings = ctx_input * math.sqrt(
            self.config.model_dim * 2
        )  # ctx_embeddings (batch size, sequence length, model dimension)
        tf_input = self.pos_encoder(ctx_embeddings)  # tf_input (batch size, sequence length, model dimension)
        context_output = self.transformer_encoder(tf_input, self.src_mask)[
            :, -1, :
        ]  # context_output (batch size, model dimension)

        return context_output


class TieredTransformer(LogModel):
    def __init__(self, config: TieredTransformerConfig):
        super().__init__(config)
        self.name = "Tiered_Transformer"
        self.config: TieredTransformerConfig = config
        self.src_mask = None
        self.log_transformer = Transformer(config)
        self.context_transformer = ContextTransformer(config)
        self.shift_window = config.shift_window

    def forward(self, src: Tensor, ctxt_vector, ctx_history, lengths=None, mask=None, has_mask=True):
        # src (num of series, batch size, sequence length, embedded dimension)
        # lengths is currently ignored, added for compatibility with LSTM-training code
        # TODO: compatibility with character level encoding
        batch_size = src.shape[1]

        if lengths is None:
            if self.log_transformer.bidirectional:
                tag_output = torch.empty((src.shape[0], src.shape[1], src.shape[2] - 2), dtype=torch.float)
            else:
                tag_output = torch.empty_like(src, dtype=torch.float)
        else:
            tag_output = torch.zeros((src.shape[0], src.shape[1], int(torch.max(lengths))), dtype=torch.float)

        tag_output = tag_output.unsqueeze(3).repeat(1, 1, 1, self.config.vocab_size)

        for idx, batch in enumerate(src):
            # batch (batch size, sequence length, embedded dimension)
            if ctxt_vector is None:
                ################ First loop without any history ##############################
                device = src.device
                ctxt_vector = torch.zeros(batch_size, self.config.context_config.model_dim).to(device)

            ################ Low level transformer ############################################
            logits, tf_hidden = self.log_transformer(
                batch, ctx_vector=ctxt_vector
            )  # (batch size, sequence length, model dimension)
            tag_output[idx][: logits.shape[0], : logits.shape[1], : logits.shape[2]] = logits

            ################ Process the output of the low level transformer ##################
            mean_hidden = torch.mean(
                tf_hidden, dim=1
            )  # mean_hidden: Mean of a low level output. (batch size, model dimension) TODO: remove this mean and see performance improvement.
            final_hidden = tf_hidden[:, -1, :]  # final_hidden: The last token step output of the low level output
            ctx_input = torch.cat(
                (mean_hidden, final_hidden), dim=1
            )  # cat_input: concatenation of mean_hidden and final_hidden (batch size, 2 * model dimension)
            unsqz_ctx_input = torch.unsqueeze(
                ctx_input, dim=1
            )  # synthetic_input: unsqueeze to concatenate with the history of a specific user. (batch size, 1, 2 * model dimension)

            if len(ctx_history.shape) == 2:
                ctx_history = unsqz_ctx_input
            else:
                ctx_history = torch.cat((unsqz_ctx_input, ctx_history), dim=1)
            # ctx_history: concatination to generate a sequence of low level outputs (batch size, history length, 2 * model dimension)
            ctx_history = ctx_history[:, -self.shift_window :, :]
            ################ Context level transformer with history #######################
            ctxt_vector = self.context_transformer(ctx_history)

        return tag_output, ctxt_vector, ctx_history  # To feed the output of
