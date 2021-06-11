# LSTM LM model
from log_analyzer.config.config import Config
from log_analyzer.model.attention import SelfAttention
from log_analyzer.config.model_config import LSTMConfig, TieredLSTMConfig
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def initialize_weights(net, initrange=1.0):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            initrange *= 1.0/np.sqrt(m.weight.data.shape[1])
            m.weight.data = initrange * \
                truncated_normal_(m.weight.data, mean=0.0, std=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            truncated_normal_(m.weight.data, mean=0.0, std=1)


class LogModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config: Config = config


class LSTMLanguageModel(LogModel):

    def __init__(self, config: LSTMConfig):
        super().__init__(config)

        self.config = config
        # Parameter setting
        self.jagged = config.jagged
        self.tiered = None

        # Layers
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.stacked_lstm = nn.LSTM(config.input_dim, config.layers[0], len(
            config.layers), batch_first=True, bidirectional=self.bidirectional)

        fc_input_dim = config.layers[-1]
        if self.bidirectional:  # If LSMTM is bidirectional, its output hidden states will be twice as large
            fc_input_dim *= 2
        # If LSTM is using attention, its hidden states will be even wider.
        if config.attention_type is not None:
            self.attention = SelfAttention(
                fc_input_dim, config.attention_dim, attention_type=config.attention_type)
            fc_input_dim *= 2
        else:
            self.attention = None

        self.hidden2tag = nn.Linear(fc_input_dim, config.vocab_size)

        # Weight initialization
        initialize_weights(self)

    @property
    def bidirectional(self):
        raise NotImplementedError(
            "Bidirectional property has to be set in child class.")

    def forward(self, sequences, lengths=None, context_vectors=None):
        # batch size, sequence length, embedded dimension
        x_lookups = self.embeddings(sequences)
        if self.tiered:
            cat_x_lookups = torch.tensor([])
            if torch.cuda.is_available():
                cat_x_lookups = cat_x_lookups.cuda()
            # x_lookups (seq len x batch x embedding)
            x_lookups = x_lookups.transpose(0, 1)
            for x_lookup in x_lookups:  # x_lookup (batch x embedding).
                # x_lookup (1 x batch x embedding)
                x_lookup = torch.unsqueeze(
                    torch.cat((x_lookup, context_vectors), dim=1), dim=0)
                # cat_x_lookups (n x batch x embedding) n = number of iteration where 1 =< n =< seq_len
                cat_x_lookups = torch.cat((cat_x_lookups, x_lookup), dim=0)
            # x_lookups (batch x seq len x embedding + context)
            x_lookups = cat_x_lookups.transpose(0, 1)

        lstm_in = x_lookups
        if self.jagged:
            lstm_in = pack_padded_sequence(
                lstm_in, lengths, enforce_sorted=False, batch_first=True)

        lstm_out, (hx, cx) = self.stacked_lstm(lstm_in)

        if self.jagged:
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)
            lstm_out = lstm_out[0]

        return lstm_out, hx


class Fwd_LSTM(LSTMLanguageModel):
    def __init__(self, config: LSTMConfig):
        self.name = "LSTM"
        super().__init__(config)

    def forward(self, sequences, lengths=None, context_vectors=None):
        lstm_out, hx = super().forward(sequences, lengths, context_vectors)

        if self.attention is not None:
            attention, _ = self.attention(lstm_out)
            output = torch.cat((lstm_out, attention.squeeze()), dim=-1)
        else:
            output = lstm_out

        tag_size = self.hidden2tag(output)

        return tag_size, lstm_out, hx

    @property
    def bidirectional(self):
        return False


class Bid_LSTM(LSTMLanguageModel):
    def __init__(self, config: LSTMConfig):
        self.name = "LSTM-Bid"
        super().__init__(config)

    def forward(self, sequences, lengths=None, context_vectors=None):
        lstm_out, hx = super().forward(sequences, lengths, context_vectors)
        # Reshape lstm_out to make forward/backward into seperate dims
        if self.jagged:
            split = lstm_out.view(
                sequences.shape[0], max(lengths), 2, lstm_out.shape[-1]//2)
        else:
            split = lstm_out.view(
                sequences.shape[0], sequences.shape[-1], 2, lstm_out.shape[-1]//2)

        # Seperate forward and backward hidden states
        forward_hidden_states = split[:, :, 0]
        backward_hidden_states = split[:, :, 1]

        # Align hidden states
        forward_hidden_states = forward_hidden_states[:, :-2]
        backward_hidden_states = backward_hidden_states[:, 2:]

        # Concat them back together
        b_f_concat = torch.cat(
            [forward_hidden_states, backward_hidden_states], -1)

        if self.attention is not None:
            attention, _ = self.attention(b_f_concat)
            b_f_concat = torch.cat((b_f_concat, attention.squeeze()), dim=-1)

        tag_size = self.hidden2tag(b_f_concat)

        return tag_size, lstm_out, hx

    @property
    def bidirectional(self):
        return True


class Context_LSTM(nn.Module):
    def __init__(self, ctxt_lv_layers, input_dim, bid):
        super().__init__()
        # Parameter setting
        self.ctxt_lv_layers = ctxt_lv_layers
        self.input_dim = input_dim

        # Layers
        self.context_lstm_layers = nn.LSTM(input_dim, self.ctxt_lv_layers[0], len(
            ctxt_lv_layers), batch_first=True)

        # Weight initialization
        initialize_weights(self)

    def forward(self, lower_lv_outputs, final_hidden, context_h, context_c, seq_len=None):

        if seq_len is not None:
            mean_hidden = torch.sum(
                lower_lv_outputs, dim=1) / seq_len.view(-1, 1)
        else:
            mean_hidden = torch.mean(lower_lv_outputs, dim=1)
        cat_input = torch.cat((mean_hidden, final_hidden[-1]), dim=1)
        synthetic_input = torch.unsqueeze(cat_input, dim=1)
        output, (context_hx, context_cx) = self.context_lstm_layers(
            synthetic_input, (context_h, context_c))

        return output, context_hx, context_cx


class Tiered_LSTM(LogModel):
    def __init__(self, config: TieredLSTMConfig):

        super().__init__(config)
        # Parameter setting
        if config.bidirectional:
            self.model = Bid_LSTM
        else:
            self.model = Fwd_LSTM

        self.bid = config.bidirectional
        low_lv_layers = config.layers

        self.ctxt_vector = None
        self.ctxt_h = None
        self.ctxt_c = None

        # Layers
        self.low_lv_lstm = self.model(config)
        self.low_lv_lstm.tiered = True  # TODO make this more elegant
        if config.bidirectional:
            input_features = low_lv_layers[-1] * 4
        else:
            input_features = low_lv_layers[-1] * 2
        self.ctxt_lv_lstm = Context_LSTM(
            config.context_layers, input_features, config.bidirectional)

        # Weight initialization
        initialize_weights(self)

    def forward(self, user_sequences, context_vectors, context_h, context_c, lengths=None):
        self.ctxt_vector = context_vectors
        self.ctxt_h = context_h
        self.ctxt_c = context_c
        if lengths is None:
            tag_output = torch.empty_like(user_sequences, dtype=torch.float)
        else:
            tag_output = torch.zeros((user_sequences.shape[0], user_sequences.shape[1], torch.max(lengths)), dtype=torch.float)

        tag_output = tag_output.unsqueeze(
                3).repeat(1, 1, 1, self.config.vocab_size)

        if torch.cuda.is_available():
            tag_output = tag_output.cuda()
        # number of steps (e.g., 3), number of users (e.g., 64), lengths of sequences (e.g., 10)
        for idx, sequences in enumerate(user_sequences):
            length = [None] if lengths is None else lengths
            tag_size, low_lv_lstm_outputs, final_hidden = self.low_lv_lstm(
                sequences, lengths=length[idx], context_vectors=self.ctxt_vector)
            if self.bid:
                final_hidden = final_hidden.view(
                    1, final_hidden.shape[1], -1)
            self.ctxt_vector, self.ctxt_h, self.ctxt_c = self.ctxt_lv_lstm(
                low_lv_lstm_outputs, final_hidden, self.ctxt_h, self.ctxt_c, seq_len=length[idx])
            tag_output[idx][:tag_size.shape[0],:tag_size.shape[1] ,:tag_size.shape[2]] = tag_size
            self.ctxt_vector = torch.squeeze(self.ctxt_vector, dim=1)
        return tag_output, self.ctxt_vector, self.ctxt_h, self.ctxt_c


if __name__ == "__main__":
    # I tried to make this code self-explanatory, but if there is any difficulty to understand ti or possible improvements, please tell me.
    test_layers = [10, 10]  # each layer has to have the same hidden units.
    test_vocab_size = 96
    test_embedding_dim = 30

    fwd_lstm_model = Fwd_LSTM(test_layers, test_vocab_size, test_embedding_dim,
                              jagged=False, tiered=False, context_vector_size=0)
    print(fwd_lstm_model)

    bid_lstm_model = Bid_LSTM(test_layers, test_vocab_size, test_embedding_dim,
                              jagged=False, tiered=False, context_vector_size=0)
    print(bid_lstm_model)
