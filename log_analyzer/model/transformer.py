"""Code related to Transformer language model"""
from log_analyzer.model.lstm import LogModel
from log_analyzer.model.model_util import initialize_weights
from log_analyzer.config.model_config import TransformerConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
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
            config.model_dim, config.attention_heads, config.feedforward_dim, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, config.layers)
        self.word_embedding = nn.Embedding(config.vocab_size, config.model_dim)

        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, lengths=None, mask=None, has_mask=True):
        # batch size, sequence length, embedded dimension
        # lengths is currently ignored, added for compatibility with LSTM-training code
        #TODO: compatibility with character level encoding
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(
                    len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        word_embeddings = self.word_embedding(src) * math.sqrt(self.config.model_dim)
        tf_input = self.pos_encoder(word_embeddings)
        tf_hidden = self.transformer_encoder(tf_input, self.src_mask)
        logits = tf_hidden @ self.word_embedding.weight.t() # word embedding encoder and decoder share weights
        # Trainer expects model to return a tuple of results (for the LSTMs this would be (lstm_out, final_hidden_state))
        # So we have to return a tuple here too (all but the first value of the tuple are discarded)
        return logits, tf_hidden # To feed the output of 
        
class Context_Transformer(LogModel):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, config: TransformerConfig):
        self.name = "Context_Transformer"
        super().__init__(config)

        self.config = config
        self.src_mask = None
        self.context_dropout = config.context_dropout
        self.context_pos_encoder = PositionalEncoding(
            config.context_model_dim, dropout=self.context_dropout)
        context_encoder_layers = nn.TransformerEncoderLayer(
            config.context_model_dim, config.context_attention_heads, 
            config.context_feedforward_dim, dropout=self.context_dropout, batch_first=True)
        self.context_transformer_encoder = nn.TransformerEncoder(
            context_encoder_layers, config.context_layers)
        self.reduce_dimension = nn.Linear(2 * config.model_dim, config.context_model_dim)

        initialize_weights(self, dist_func=nn.init.xavier_uniform_)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, ctx_history, lengths=None, mask=None, has_mask=True):

        if has_mask:
            device = ctx_history.device
            if self.src_mask is None or self.src_mask.shape[-1] != ctx_history.shape[-1]:
                mask = self._generate_square_subsequent_mask(ctx_history.shape[-1]).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        ctx_input = self.reduce_dimension(ctx_history) # ctx_input (batch size, sequence length, 2 * model dimension)
        ctx_embeddings = ctx_input * math.sqrt(self.config.context_model_dim * 2) # ctx_embeddings (batch size, sequence length, model dimension)
        tf_input = self.context_pos_encoder(ctx_embeddings) # tf_input (batch size, sequence length, model dimension)
        context_output = self.context_transformer_encoder(tf_input, self.src_mask)[:,-1,:] # context_output (batch size, model dimension)

        return context_output 

class Tiered_Transformer(LogModel):

    def __init__(self, config: TransformerConfig):
        self.name = "Tiered_Transformer"
        self.config = config
        self.src_mask = None
        self.log_transformer = Transformer(config)
        self.context_transformer = Context_Transformer(config)
        self.ctxt_vector = None
        self.ctx_history = torch.Tensor([])
        
    def forward(self, src, ctx_history, lengths=None, mask=None, has_mask=True):
        # src (num of series, batch size, sequence length, embedded dimension)
        # lengths is currently ignored, added for compatibility with LSTM-training code
        #TODO: compatibility with character level encoding
        batch_size = src.shape[1]
        
        for batch in src:
        # batch (batch size, sequence length, embedded dimension)
            if ctx_history is None:
                ################ First loop without any history ##############################
                self.ctxt_vector = torch.zeros(batch_size, self.config.context_model_dim)
            else:
                ################ Context level transformer with history #######################
                self.ctxt_vector = self.context_transformer(ctx_history)

            ################ Low level transformer ############################################
            logits, tf_hidden = self.log_transformer(batch, ctx_vector = self.ctxt_vector) 
            
            ################ Process the output of the low level transformer ##################
            mean_hidden = torch.mean(tf_hidden, dim=1)            # mean_hidden: Mean of a low level output. 
            final_hidden = tf_hidden[:,-1,:]                      # final_hidden: The last time step output of the low level output
            ctx_input = torch.cat((mean_hidden, final_hidden), dim=1) # cat_input: concatenation of mean_hidden and final_hidden (batch size, 2 * model dimension) 
            unsqz_ctx_input = torch.unsqueeze(ctx_input, dim=1)       # synthetic_input: unsqueeze to concatenate with the history of a specific user. (batch size, 1, 2 * model dimension) 
            ctx_history = torch.cat((unsqz_ctx_input, ctx_input), dim=1) # ctx_history: concatination to generate a sequence of low level outputs (batch size, history length, 2 * model dimension)

        return logits, ctx_history # To feed the output of 