"""Code related to Transformer language model."""
import math

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
        if self.name == "Transformer_Decoder" :
            decoder_layers = nn.TransformerDecoderLayer(
                self.model_dim, self.attention_heads, self.feedforward_dim, dropout=self.dropout, batch_first=True
                )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, self.layers)
        else:
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

    def forward(self, src, ctx_vector=None, lengths=None, mask=None, targets=None):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code
        # TODO: compatibility with character level encoding

        word_embeddings = self.word_embedding(src) * math.sqrt(self.config.model_dim)
        if ctx_vector is not None:
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
            loss, _ = self.compute_loss(logits, targets, lengths, mask)

        if self.tiered:
            return logits, tf_hidden, loss
        return logits, loss

class TieredTransformer(TieredLogModel):
    def __init__(self, config: TieredTransformerConfig):
        super().__init__(config)
        self.config: TransformerConfig = config
        self.bidirectional = False 
        self.dropout = config.dropout
        self.model_dim = config.model_dim
        self.layers = config.layers
        self.attention_heads = config.attention_heads
        self.feedforward_dim = config.feedforward_dim
        self.vocab_size = config.vocab_size
        self.name = "Tiered_Transformer"
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout)
        self.word_embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.transformer_model = nn.Transformer(d_model=self.config.model_dim,
                                            nhead=self.config.attention_heads, 
                                            num_encoder_layers=self.config.layers,
                                            num_decoder_layers=self.config.context_config.layers,
                                            dim_feedforward =self.config.feedforward_dim,
                                            dropout=self.dropout,
                                            batch_first=True)
        self.shift_window = config.shift_window

        # User model state
        self.context_model_dim = config.context_config.model_dim
        self.context_input_dimension = config.input_dim
        self.shift_window = config.shift_window
        self.n_users = 1200
        self.saved_context_histories = torch.zeros([self.n_users, self.shift_window, self.model_dim])
        self.saved_context_history_lengths = torch.zeros([self.n_users], dtype=torch.int16)

        initialize_weights(self, dist_func=nn.init.xavier_uniform_)
    
    def gen_mask(self, seq_len, device=None, mask=None, has_mask=True):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code
        if has_mask:
            mask = _generate_square_subsequent_mask(seq_len).to(device)
        else:
            mask = None
        return mask
        
    def gen_pad_mask(self, ctx_history, history_length, device= None, has_mask=True):
        history_padding_length = ctx_history.shape[1] - history_length.long() 
        if has_mask:
            mask = torch.ones([ctx_history.shape[0], ctx_history.shape[1]]) != 1
            for p, i in zip(history_padding_length, range(mask.shape[0])):
                mask[i,:p] = True
        else:
            mask = None
        return mask.to(device)

    def forward(self, src: Tensor, lengths=None, mask=None, targets=None):
        users, src = src
        # Convert users list to python list
        users = [user.item() for user in users]

        # Get the state for the users in the batch
        context_history, history_length = self.get_batch_data(users, src.device)

        # Get the number of steps in the batch
        self.num_steps = src.shape[0]
        batch_size = src.shape[1]

        token_output = torch.empty_like(src, dtype=torch.float)

        token_output = token_output.unsqueeze(3).repeat(1, 1, 1, self.config.vocab_size)
        
        for idx, batch in enumerate(src):
            tgt_mask = self.gen_mask(batch.shape[-1], device=src.device)
            src_pad_mask = self.gen_pad_mask(context_history, history_length, device=src.device) 
            embedding_tgt_input = self.word_embedding(batch)  * math.sqrt(self.model_dim)

            src_input = self.pos_encoder(context_history)
            tgt_input = self.pos_encoder(embedding_tgt_input)
            
            tf_hidden = self.transformer_model(src = src_input,
                                            src_key_padding_mask = src_pad_mask, 
                                            tgt = tgt_input,
                                            tgt_mask = tgt_mask)
            tf_hidden_mean = torch.unsqueeze(torch.mean(tf_hidden, dim = 1), dim=1) * math.sqrt(self.model_dim)
            context_history = torch.cat([context_history, tf_hidden_mean], dim =1)[:, -self.shift_window:, :]
            logits = tf_hidden @ self.word_embedding.weight.t()
            token_output[idx][: logits.shape[0], : logits.shape[1], : logits.shape[2]] = logits
            history_length = torch.min(history_length + 1, torch.ones(history_length.shape, dtype=int)*self.shift_window)
        # Update context state
        self.update_state(users, context_history)

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, _ = self.compute_loss(token_output, targets, lengths, mask)
        return token_output, loss

    def get_batch_data(self, users, device):
        """Given a list of users, fetch the relevant history and model data for
        each user."""
        history = self.saved_context_histories[torch.tensor(users)]
        history_lengths = self.saved_context_history_lengths[torch.tensor(users)]
        # Crop the length of history returned to max history_length amongst users in this batch
        max_length = torch.max(history_lengths)
        return history[:, -max_length:, :].to(device), history_lengths

    def update_state(self, users, context_history):
        """Given one batch of history/model data output by the model, update
        the stored state for future use."""
        context_history = context_history.detach().cpu()
        for user in users:
            self.saved_context_history_lengths[user] = min(
                self.saved_context_history_lengths[user] + self.num_steps, self.shift_window
            )
        max_length = torch.max(self.saved_context_history_lengths[torch.tensor(users)])
        self.saved_context_histories[torch.tensor(users), -max_length:, :] = context_history[:, -max_length:, :]
