# LSTM LM model

from typing import Type

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
    def __init__(self, config: Config):
        super().__init__()
        self.config: Config = config


class LSTMLanguageModel(LogModel):
    def __init__(self, config: LSTMConfig):
        super().__init__(config)

        self.config = config
        # Parameter setting
        self.tiered: bool = False

        # Layers
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.stacked_lstm = nn.LSTM(
            config.input_dim,
            config.layers[0],
            len(config.layers),
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        fc_input_dim = config.layers[-1]
        if self.bidirectional:  # If LSMTM is bidirectional, its output hidden states will be twice as large
            fc_input_dim *= 2
        # If LSTM is using attention, its hidden states will be even wider.
        if config.attention_type is not None:
            if config.sequence_length is not None:
                seq_len = config.sequence_length
                seq_len = seq_len - 2 if self.bidirectional else seq_len
            else:
                seq_len = None
            self.attention = SelfAttention(
                fc_input_dim,
                config.attention_dim,
                attention_type=config.attention_type,
                seq_len=seq_len,
            )
            fc_input_dim *= 2
            self.has_attention = True
        else:
            self.has_attention = False

        self.hidden2tag = nn.Linear(fc_input_dim, config.vocab_size)

        # Weight initialization
        initialize_weights(self)

    @property
    def bidirectional(self):
        raise NotImplementedError("Bidirectional property has to be set in child class.")

    def forward(self, sequences, lengths: Tensor = None, context_vectors=None, mask=None):
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
                lengths = lengths.squeeze()
            lstm_in = pack_padded_sequence(lstm_in, lengths.cpu(), enforce_sorted=False, batch_first=True)

        lstm_out, (hx, cx) = self.stacked_lstm(lstm_in)

        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        return lstm_out, hx


class FwdLSTM(LSTMLanguageModel):
    def __init__(self, config: LSTMConfig):
        self.name = "LSTM"
        super().__init__(config)

    def forward(self, sequences, lengths=None, context_vectors=None, mask=None):
        lstm_out, hx = super().forward(sequences, lengths, context_vectors)

        if self.has_attention:
            attention, _ = self.attention(lstm_out, mask)
            output = torch.cat((lstm_out, attention), dim=-1)
        else:
            output = lstm_out

        tag_size = self.hidden2tag(output)

        return tag_size, lstm_out, hx

    @property
    def bidirectional(self):
        return False


class BidLSTM(LSTMLanguageModel):
    def __init__(self, config: LSTMConfig):
        self.name = "LSTM-Bid"
        super().__init__(config)

    def forward(self, sequences: torch.Tensor, lengths=None, context_vectors=None, mask=None):
        lstm_out, hx = super().forward(sequences, lengths, context_vectors)
        # Reshape lstm_out to make forward/backward into seperate dims

        if lengths is not None:
            split = lstm_out.view(sequences.shape[0], max(lengths), 2, lstm_out.shape[-1] // 2)
        else:
            split = lstm_out.view(sequences.shape[0], sequences.shape[-1], 2, lstm_out.shape[-1] // 2)

        # Seperate forward and backward hidden states
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

        tag_size = self.hidden2tag(b_f_concat)

        return tag_size, lstm_out, hx

    @property
    def bidirectional(self):
        return True


class ContextLSTM(nn.Module):
    def __init__(self, ctxt_lv_layers, input_dim):
        super().__init__()
        # Parameter setting
        self.ctxt_lv_layers = ctxt_lv_layers
        self.input_dim = input_dim

        # Layers
        self.context_lstm_layers = nn.LSTM(input_dim, self.ctxt_lv_layers[0], len(ctxt_lv_layers), batch_first=True)

        # Weight initialization
        initialize_weights(self)

    def forward(self, lower_lv_outputs, final_hidden, context_h, context_c, seq_len=None):

        if seq_len is not None:
            mean_hidden = torch.sum(lower_lv_outputs, dim=1) / seq_len.view(-1, 1)
        else:
            mean_hidden = torch.mean(lower_lv_outputs, dim=1)
        cat_input = torch.cat((mean_hidden, final_hidden[-1]), dim=1)
        synthetic_input = torch.unsqueeze(cat_input, dim=1)
        output, (context_hx, context_cx) = self.context_lstm_layers(synthetic_input, (context_h, context_c))

        return output, context_hx, context_cx


class TieredLSTM(LogModel):
    def __init__(self, config: TieredLSTMConfig, bidirectional):

        super().__init__(config)
        # Parameter setting
        self.model: Type[LSTMLanguageModel]
        if bidirectional:
            self.model = BidLSTM
        else:
            self.model = FwdLSTM

        low_lv_layers = config.layers

        self.ctxt_vector = None
        self.ctxt_h = None
        self.ctxt_c = None

        # Layers
        self.low_lv_lstm = self.model(config)
        self.low_lv_lstm.tiered = True  # TODO make this more elegant
        if bidirectional:
            input_features = low_lv_layers[-1] * 4
        else:
            input_features = low_lv_layers[-1] * 2
        self.ctxt_lv_lstm = ContextLSTM(config.context_layers, input_features)

        # Weight initialization
        initialize_weights(self)

    def forward(self, user_sequences, context_vectors, context_h, context_c, lengths=None):
        self.ctxt_vector = context_vectors
        self.ctxt_h = context_h
        self.ctxt_c = context_c
        if lengths is None:
            if self.low_lv_lstm.bidirectional:
                tag_output = torch.empty(
                    (
                        user_sequences.shape[0],
                        user_sequences.shape[1],
                        user_sequences.shape[2] - 2,
                    ),
                    dtype=torch.float,
                )
            else:
                tag_output = torch.empty_like(user_sequences, dtype=torch.float)
        else:
            tag_output = torch.zeros(
                (user_sequences.shape[0], user_sequences.shape[1], torch.max(lengths)),
                dtype=torch.float,
            )

        tag_output = tag_output.unsqueeze(3).repeat(1, 1, 1, self.config.vocab_size)

        if Application.instance().using_cuda:
            tag_output = tag_output.cuda()
        # number of steps (e.g., 3), number of users (e.g., 64), lengths of
        # sequences (e.g., 10)
        for idx, sequences in enumerate(user_sequences):
            length = None if lengths is None else lengths[idx]
            tag_size, low_lv_lstm_outputs, final_hidden = self.low_lv_lstm(
                sequences, lengths=length, context_vectors=self.ctxt_vector
            )
            if self.low_lv_lstm.bidirectional:
                final_hidden = final_hidden.view(1, final_hidden.shape[1], -1)
            self.ctxt_vector, self.ctxt_h, self.ctxt_c = self.ctxt_lv_lstm(
                low_lv_lstm_outputs,
                final_hidden,
                self.ctxt_h,
                self.ctxt_c,
                seq_len=length,
            )
            tag_output[idx][: tag_size.shape[0], : tag_size.shape[1], : tag_size.shape[2]] = tag_size
            self.ctxt_vector = torch.squeeze(self.ctxt_vector, dim=1)
        return tag_output, self.ctxt_vector, self.ctxt_h, self.ctxt_c


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
