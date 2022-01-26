# LSTM log language model

from typing import Optional, Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from log_analyzer.application import Application
from log_analyzer.config.config import Config
from log_analyzer.config.model_config import LSTMConfig, TieredLSTMConfig
from log_analyzer.model.attention import SelfAttention
from log_analyzer.model.model_util import initialize_weights


class LogModel(nn.Module):
    """Superclass for all log-data language models.
    
    All log-data language models should implement the forward() function with:
    
    input: input_sequence, model_info, lengths=None, mask=None, targets=None
    output: output_sequence, model_info, loss=None
    
    Where model_info is any extra info needed by that model (e.g. context info, history)
    either as a singular value or a tuple of values
    Loss should be returned if targets is provided, otherwise None is returned."""

    def __init__(self, config: Config):
        super().__init__()
        self.config: Config = config
        self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=0)

    def compute_loss(self, output: torch.Tensor, Y, lengths, mask: torch.Tensor):
        """Computes the loss for the given model output and ground truth."""
        token_losses = self.criterion(output.transpose(1, 2), Y)
        if mask is not None:
            token_losses = token_losses * mask
        line_losses = torch.mean(token_losses, dim=1)
        loss = torch.mean(line_losses, dim=0)

        # Return the loss, as well as extra details like loss per line
        return loss, line_losses


class TieredLogModel(LogModel):
    """Superclass for tiered log-data language models."""

    def __init__(self, config: Config):
        super().__init__(config)

    def compute_loss(self, output: Tensor, Y: Tensor, lengths, mask):
        """Computes the loss for the given model output and ground truth."""
        loss = torch.tensor(0.0)
        line_losses_list = torch.empty(output.shape[:-2], dtype=torch.float)
        if Application.instance().using_cuda:
            line_losses_list = line_losses_list.cuda()
        if lengths is not None:
            max_length = int(torch.max(lengths))
        # output (num_steps x batch x length x embedding dimension)  Y
        # (num_steps x batch x length)
        for i, (step_output, step_y) in enumerate(zip(output, Y)):
            # On notebook, I checked it with forward LSTM and word
            # tokenization. Further checks have to be done...
            if lengths is not None:
                token_losses = self.criterion(step_output.transpose(1, 2), step_y[:, :max_length])
                masked_losses = token_losses * mask[i][:, :max_length]
                line_losses = torch.sum(masked_losses, dim=1)
            else:
                token_losses = self.criterion(step_output.transpose(1, 2), step_y)
                line_losses = torch.mean(token_losses, dim=1)
            line_losses_list[i] = line_losses
            step_loss = torch.mean(line_losses, dim=0)
            loss += step_loss
        loss /= len(Y)
        return loss, line_losses_list


class LSTMLanguageModel(LogModel):
    """Superclass for non-tiered LSTM log-data language models"""

    def __init__(self, config: LSTMConfig):
        super().__init__(config)

        self.config = config
        # Parameter setting
        self.tiered: bool = False

        # Layers
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.stacked_lstm = nn.LSTM(
            config.input_dim, config.layers[0], len(config.layers), batch_first=True, bidirectional=self.bidirectional,
        )

        fc_input_dim = config.layers[-1]
        if self.bidirectional:  # If LSMTM is bidirectional, its output hidden states will be twice as large
            fc_input_dim *= 2
        # If LSTM is using attention, its hidden states will be even wider.
        seq_len: Optional[int]
        if config.attention_type is not None:
            if config.sequence_length is not None:
                seq_len = config.sequence_length
                seq_len = seq_len - 2 if self.bidirectional else seq_len
            else:
                seq_len = None
            self.attention = SelfAttention(
                fc_input_dim, config.attention_dim, attention_type=config.attention_type, seq_len=seq_len,
            )
            fc_input_dim *= 2
            self.has_attention = True
        else:
            self.has_attention = False

        self.get_token_output = nn.Linear(fc_input_dim, config.vocab_size)

        # Weight initialization
        initialize_weights(self)

    @property
    def bidirectional(self):
        raise NotImplementedError("Bidirectional property has to be set in child class.")

    def forward(self, sequences, lengths: Tensor = None, context_vectors=None, mask=None):
        """Performs token embedding, context-prepending if model is tiered, and runs the LSTM on the input sequences."""
        # batch size, sequence length, embedded dimension
        x_lookups = self.embeddings(sequences)
        if self.tiered:
            cat_x_lookups = torch.Tensor([])
            if Application.instance().using_cuda:
                cat_x_lookups = cat_x_lookups.cuda()
            # x_lookups (seq len x batch x embedding)
            x_lookups = x_lookups.transpose(0, 1)
            for x_lookup in x_lookups:  # x_lookup (batch x embedding).
                # x_lookup (1 x batch x embedding)
                x_lookup = torch.unsqueeze(torch.cat((x_lookup, context_vectors), dim=1), dim=0)
                # cat_x_lookups (n x batch x embedding) n = number of iteration
                # where 1 =< n =< seq_len
                cat_x_lookups = torch.cat((cat_x_lookups, x_lookup), dim=0)
            # x_lookups (batch x seq len x embedding + context)
            x_lookups = cat_x_lookups.transpose(0, 1)

        lstm_in = x_lookups

        if lengths is not None:
            if len(lengths.shape) > 1:
                lengths = lengths.squeeze(1)
            lstm_in = pack_padded_sequence(lstm_in, lengths.cpu(), enforce_sorted=False, batch_first=True)

        lstm_out, (hx, cx) = self.stacked_lstm(lstm_in)

        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        return lstm_out, hx


class FwdLSTM(LSTMLanguageModel):
    """Standard forward LSTM model."""

    def __init__(self, config: LSTMConfig):
        self.name = "LSTM"
        super().__init__(config)

    def forward(self, sequences, lengths=None, context_vectors=None, mask=None, targets=None):
        """Handles attention (if relevant) and grabs the final token output guesses.
        
        Returns: predicted_tokens, (lstm_output_features, final_hidden_state), loss
        If targets is None then loss is returned as None"""
        lstm_out, hx = super().forward(sequences, lengths, context_vectors)

        if self.has_attention:
            attention, _ = self.attention(lstm_out, mask)
            output = torch.cat((lstm_out, attention), dim=-1)
        else:
            output = lstm_out

        token_output = self.get_token_output(output)

        if targets is not None:
            # Compute and return loss if targets is given
            loss = self.compute_loss(token_output, targets, lengths, mask)
            return token_output, (lstm_out, hx), loss

        return token_output, (lstm_out, hx), None

    @property
    def bidirectional(self):
        return False


class BidLSTM(LSTMLanguageModel):
    """Standard bidirectional LSTM model."""

    def __init__(self, config: LSTMConfig):
        self.name = "LSTM-Bid"
        super().__init__(config)

    def forward(self, sequences: torch.Tensor, lengths=None, context_vectors=None, mask=None, targets=None):
        """Handles bidir-state-alignment, attention (if relevant) and grabs the final token output guesses.
        
        Returns: predicted_tokens, (lstm_output_features, final_hidden_state), loss
        If targets is None then loss is returned as None"""
        lstm_out, hx = super().forward(sequences, lengths, context_vectors)
        # Reshape lstm_out to make forward/backward into separate dims

        if lengths is not None:
            split = lstm_out.view(sequences.shape[0], max(lengths), 2, lstm_out.shape[-1] // 2)
        else:
            split = lstm_out.view(sequences.shape[0], sequences.shape[-1], 2, lstm_out.shape[-1] // 2)

        # Separate forward and backward hidden states
        forward_hidden_states = split[:, :, 0]
        backward_hidden_states = split[:, :, 1]

        # Align hidden states
        forward_hidden_states = forward_hidden_states[:, :-2]
        backward_hidden_states = backward_hidden_states[:, 2:]

        # Concat them back together
        b_f_concat = torch.cat([forward_hidden_states, backward_hidden_states], -1)

        if self.has_attention:
            attention, _ = self.attention(b_f_concat, mask)
            b_f_concat = torch.cat((b_f_concat, attention.squeeze()), dim=-1)

        token_output = self.get_token_output(b_f_concat)

        if targets is not None:
            # Compute and return loss if targets is given
            loss = self.compute_loss(token_output, targets, lengths, mask)
            return token_output, (lstm_out, hx), loss

        return token_output, (lstm_out, hx), None

    @property
    def bidirectional(self):
        return True


class ContextLSTM(nn.Module):
    """High-level context LSTM model used in the tiered-LSTM models."""

    def __init__(self, ctxt_lv_layers, input_dim):
        super().__init__()
        # Parameter setting
        self.ctxt_lv_layers = ctxt_lv_layers
        self.input_dim = input_dim

        # Layers
        self.context_lstm_layers = nn.LSTM(input_dim, self.ctxt_lv_layers[0], len(ctxt_lv_layers), batch_first=True)

        # Weight initialization
        initialize_weights(self)

    def forward(self, lower_lv_outputs, model_info, seq_len=None):
        """Handles processing and updating of context info.
        
        Returns: context_output, (final_hidden_state, final_cell_state)"""
        final_hidden, context_h, context_c = model_info

        if seq_len is not None:
            mean_hidden = torch.sum(lower_lv_outputs, dim=1) / seq_len.view(-1, 1)
        else:
            mean_hidden = torch.mean(lower_lv_outputs, dim=1)
        cat_input = torch.cat((mean_hidden, final_hidden[-1]), dim=1)
        synthetic_input = torch.unsqueeze(cat_input, dim=1)
        output, (context_hx, context_cx) = self.context_lstm_layers(synthetic_input, (context_h, context_c))

        return output, (context_hx, context_cx)


class TieredLSTM(TieredLogModel):
    """Tiered-LSTM model, combines a standard forward or bidirectional LSTM model for
    log-level analysis and a context LSTM for propagation of high-level context information."""

    def __init__(self, config: TieredLSTMConfig, bidirectional):

        super().__init__(config)
        # Parameter setting
        self.model: Type[LSTMLanguageModel]
        if bidirectional:
            self.model = BidLSTM
        else:
            self.model = FwdLSTM

        low_lv_layers = config.layers

        self.context_vector = None
        self.context_hidden_state = None
        self.context_cell_state = None

        # Layers
        self.low_lv_lstm = self.model(config)
        self.low_lv_lstm.tiered = True  # TODO make this more elegant
        if bidirectional:
            input_features = low_lv_layers[-1] * 4
        else:
            input_features = low_lv_layers[-1] * 2
        self.context_lstm = ContextLSTM(config.context_layers, input_features)

        # Weight initialization
        initialize_weights(self)

    def forward(self, user_sequences, model_info, lengths=None, targets=None):
        """Forward pass of tiered LSTM model.
        
        Returns: predicted_tokens, (context_vector, final_context_hidden_state, final_context_cell_state), loss
        If targets is None then loss is returned as None"""
        self.context_vector = model_info[0]
        self.context_hidden_state = model_info[1]
        self.context_cell_state = model_info[2]

        if self.low_lv_lstm.bidirectional:
            token_output = torch.empty(
                (user_sequences.shape[0], user_sequences.shape[1], user_sequences.shape[2] - 2,), dtype=torch.float,
            )
        else:
            token_output = torch.empty_like(user_sequences, dtype=torch.float)

        token_output = token_output.unsqueeze(3).repeat(1, 1, 1, self.config.vocab_size)

        if Application.instance().using_cuda:
            token_output = token_output.cuda()
        # number of steps (e.g., 3), number of users (e.g., 64), lengths of
        # sequences (e.g., 10)
        for idx, sequences in enumerate(user_sequences):
            length = None if lengths is None else lengths[idx]
            tag_size, (low_lv_lstm_outputs, final_hidden), _ = self.low_lv_lstm(
                sequences, lengths=length, context_vectors=self.context_vector
            )
            if self.low_lv_lstm.bidirectional:
                final_hidden = final_hidden.view(1, final_hidden.shape[1], -1)
            self.context_vector, (self.context_hidden_state, self.context_cell_state) = self.context_lstm(
                low_lv_lstm_outputs, (final_hidden, self.context_hidden_state, self.context_cell_state), seq_len=length,
            )
            token_output[idx][: tag_size.shape[0], : tag_size.shape[1], : tag_size.shape[2]] = tag_size
            self.context_vector = torch.squeeze(self.context_vector, dim=1)

        if targets is not None:
            # Compute and return loss if targets is given
            loss = self.compute_loss(token_output, targets, lengths)
            return token_output, (self.context_vector, self.context_hidden_state, self.context_cell_state), loss

        return token_output, (self.context_vector, self.context_hidden_state, self.context_cell_state), None


# if __name__ == "__main__":
# I tried to make this code self-explanatory,
# but if there is any difficulty to understand ti or possible improvements, please tell me.
# test_layers = [10, 10]  # each layer has to have the same hidden units.
# test_vocab_size = 96
# test_embedding_dim = 30

# fwd_lstm_model = Fwd_LSTM(test_layers, test_vocab_size, test_embedding_dim,
#                          tiered=False, context_vector_size=0)
# print(fwd_lstm_model)

# bid_lstm_model = Bid_LSTM(test_layers, test_vocab_size, test_embedding_dim,
#                          tiered=False, context_vector_size=0)
# print(bid_lstm_model)
