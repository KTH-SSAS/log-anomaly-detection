"""Code related to Transformer language model."""
import math
from abc import abstractmethod
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from log_analyzer.application import Application
from log_analyzer.config.model_config import MultilineTransformerConfig, TieredTransformerConfig, TransformerConfig
from log_analyzer.model.lstm import LogModel, MultilineLogModel, TieredLogModel
from log_analyzer.model.model_util import initialize_weights


def _generate_square_subsequent_mask(seq_len):
    """Generates a standard square subsequent mask for self-attention."""
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def _generate_multiline_asymmetric_mask(source_length, key_value_length):
    """Generates an asymmetric mask for self-attention where source length is
    larger than the key/value (aka context) length.

    For example, a multiline transformer with window_size 5 will have:
    source length = 5
    key_value_length = 9
    Mask:
    0 0 0 0 0 i i i i
    i 0 0 0 0 0 i i i
    i i 0 0 0 0 0 i i
    i i i 0 0 0 0 0 i
    i i i i 0 0 0 0 0
    where i=-inf
    """
    # Initialise mask of ones - key_value_length wide, source_length tall
    mask = torch.ones(key_value_length, source_length)
    # We want to fill in a source_length wide diagonal with 0
    for row in range(key_value_length):
        for col in range(row, row + key_value_length):
            mask[row, col] = 0
    # Replace the ones with -inf
    mask = mask.float().masked_fill(mask == 1, float("-inf"))
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
    def forward(self, sequences, lengths: Tensor = None, mask=None, targets=None):
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
        
        # Don't compute loss for the first token (source_user) when including timestamp
        if self.config.include_timestamp:
            logits = logits[:, 1:, :]
            targets = targets[:, 1:]

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
        for _ in range(self.n_users):
            self.ctx_histories.append(torch.zeros([1, self.ctx_dim]))
        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

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
        ctx_histories = self.get_ctx_data(users, tgt.device)

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
            src_mask = _generate_square_subsequent_mask(src_input.shape[1]).to(src_input.device)
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
                ctx_histories = [torch.zeros([1, self.ctx_dim]).to(tgt.device) for _ in users]
            logits = tf_hidden @ self.word_embedding.weight.t()
            token_output[idx][: logits.shape[0], : logits.shape[1], : logits.shape[2]] = logits
        # Update context state
        self.update_ctx_data(users, ctx_histories)

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, _ = self.compute_loss(token_output, targets)
        return token_output, loss

    def get_ctx_data(self, users, device):
        """Given a list of users, fetch the relevant history and model data for
        each user."""
        history = []
        for u in users:
            history.append(self.ctx_histories[u].to(device))
        return history

    def update_ctx_data(self, users, ctx_histories):
        for u, ctx_history in zip(users, ctx_histories):
            self.ctx_histories[u] = ctx_history.detach().cpu()


class SKVTransformerEncoderLayer(nn.Module):
    r"""Near-duplicate of PyTorch's nn.TransformerEncoderLayer to redefine forward function.

    Purpose: allow separate query/key/value tensors as input.

    See torch.nn.modules.transformer.TransformerEncoderLayer docs for more information."""
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = torch.functional.F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        key: Tensor,
        value: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Redefined forward function to allow separate query/key/value tensors.

        see the nn.TransformerEncoderLayer docs for more information.
        """
        src2 = self.self_attn(src, key, value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SKVTransformerEncoder(nn.Module):
    r"""Near-duplicate of PyTorch's nn.TransformerEncoder to allow passing
    separate source(aka query)/key/value tensors as input.

    See torch.nn.modules.transformer.TransformerEncoder docs for more information."""
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Redefined forward function to allow separate query/key/value tensors.

        see the nn.TransformerEncoder docs for more information.
        """
        output = src

        for mod in self.layers:
            if isinstance(mod, SKVTransformerEncoderLayer):
                output = mod(output, key, value, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MultilineTransformer(MultilineLogModel):
    """Transformer that works across multiple log lines - each "token" input is a single log line.

    The type of sentence embedding used is defined in the config. Valid options are currently:
    "mean": embeds words to the full model_dim, then takes the element-wise mean of the words in the log line.
                   in this case loss is computed in the embedding space (since mean is not reversible)
    "stack": embeds words to model_dim/sentence_length dims, then stacks them to form sentence embedding.
                   in this case loss is computed for each word prediction (since concatenation is reversible)

    Output: predicted embedding value for the next logline."""

    def __init__(self, config: MultilineTransformerConfig, bidirectional):
        super().__init__(config)

        self.bidirectional = bidirectional
        # Currently no support for bidirectional multiline transformer
        self.bidirectional = False
        self.config: MultilineTransformerConfig = config
        self.name = "Multiline Transformer"
        self.src_mask = None

        self.dropout = config.dropout
        self.model_dim = config.model_dim
        self.layers = config.layers
        self.attention_heads = config.attention_heads
        self.feedforward_dim = config.feedforward_dim
        self.vocab_size = config.vocab_size
        # Window size for the purposes of pe, etc.
        self.virtual_shift_window = self.config.shift_window * 2 - 1

        self.using_cuda = Application.instance().using_cuda

        # Prepare the sentence embedding
        if self.config.sentence_embedding == "mean":
            self._sentence_embedding = partial(torch.mean, dim=2)
            self._sentence_deembedding = None  # Cannot reverse mean()
            embedding_dim = self.model_dim
        elif self.config.sentence_embedding == "stack":
            # In this case words will be embedded to self.model_dim/sentence_length dims, then stackd to form
            # a self.model_dim long sentence embedding
            # Loss function will be cross entropy
            embedding_dim = int(self.model_dim / 10)
            assert (
                embedding_dim == self.model_dim / 10
            ), "For 'stack' sentence embedding, model_dim must be divisible by 10"
            self._sentence_embedding = partial(torch.flatten, start_dim=2)
            self._sentence_deembedding = lambda t: t.reshape(t.shape[0], t.shape[1], 10, embedding_dim)
            self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=0)

        # Check if we have pretrained embedding weights to use
        if self.config.embeddings_path not in ("", None):
            embedding_weights = torch.load(self.config.embeddings_path)
            embedding_weights = embedding_weights["word_embedding.weight"]
            # Load the pretrained embedding weights, and freeze them
            self._word_embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
            # Move the word embeddings to the correct device - pretrained might be from CPU or GPU!
            if self.using_cuda:
                self._word_embedding.cuda()
            else:
                self._word_embedding.cpu()
        else:
            # Normal, learnable embeddings
            self._word_embedding = nn.Embedding(self.vocab_size, embedding_dim)

        self.pos_encoder = PositionalEncoding(self.model_dim, dropout=self.dropout, max_len=self.virtual_shift_window)
        encoder_layers = SKVTransformerEncoderLayer(
            self.model_dim, self.attention_heads, self.feedforward_dim, dropout=self.dropout, batch_first=True
        )
        self.transformer_encoder = SKVTransformerEncoder(encoder_layers, self.layers)

        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def get_mask(self, src: torch.Tensor):
        # batch size, sequence length, embedded dimension
        seq_len = src.shape[1]
        device = src.device
        # Simple case - input sequences are shift_window long, generate standard self-attention mask
        if self.src_mask is None or self.src_mask.shape[0] != seq_len:
            if seq_len == self.config.shift_window:
                mask = _generate_square_subsequent_mask(seq_len).to(device)
            else:
                # We have extra history so that each step can have the full shift_window history length.
                # We must generate an appropriate mask so each step has exactly shift_window long history
                # This mask will **not** be square (as the length of source (i.e. the length of the output)
                # is not the same as the length of the context).
                # E.g. , with shift_window of 3, this will be a 3x5 matrix like (i=-inf):
                # 0 0 0 i i
                # i 0 0 0 i
                # i i 0 0 0
                mask = _generate_multiline_asymmetric_mask(seq_len, self.config.shift_window).to(device)
            self.src_mask = mask

        return self.src_mask

    def word_embedding(self, src):
        """Performs word embedding, i.e. from word token to n-dimensional
        vector representation."""
        return self._word_embedding(src)

    def sentence_embedding(self, src):
        """Performs sentence embedding, taking a sequence of embedded word
        tokens and producing a singular 'sentence token'."""
        return self._sentence_embedding(src)

    def sentence_deembedding(self, src):
        """Reverses sentence embedding (if possible), taking a sentence token
        and yielding a sequence of embedded word tokens."""
        return self._sentence_deembedding(src)

    def forward(self, sequences: Tensor, lengths=None, mask=None, targets: Tensor = None):
        # sequences: (batch, sequence, log_line)
        # Step 1: Use sentence embedding to summarise each logline as a single token
        # Step 2: Apply transformer across this sequence of logline tokens

        # Prepare mask
        self.src_mask = self.get_mask(sequences)

        # Apply word embedding to each log line in each sequence in each batch
        word_embeddings = self.word_embedding(sequences)
        # word_embeddings: (batch size, sequence length, logline length, embedded dimension)
        line_embeddings = self.sentence_embedding(
            word_embeddings
        )  # Logline embeddings - average of word tokens in the line
        # line_embeddings: (batch size, sequence_length, sentence embedded dimension)

        # Add positional encoding to the line embeddings
        line_embeddings = self.pos_encoder(line_embeddings)

        # Extract the src (lines producing output) from the line embeddings, to be input as source (aka query) to
        # self attention (the full sequence will be used for the key+value pairs, i.e. what is attended over)
        src_line_embeddings = line_embeddings[:, self.config.shift_window - 1 :, :]

        # src_key_padding_mask: if provided, specified padding elements in the key will be ignored by the attention.
        pad_mask = mask == 0 if mask is not None else None

        tf_hidden = self.transformer_encoder(
            src_line_embeddings, line_embeddings, line_embeddings, self.src_mask, src_key_padding_mask=pad_mask
        )

        # Try to reverse sentence embedding to produce logits
        if self._sentence_deembedding is None:
            logits = tf_hidden
        else:
            logits = self.sentence_deembedding(tf_hidden) @ self._word_embedding.weight.t()

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, _ = self.compute_loss(logits, targets)

        return logits, loss
