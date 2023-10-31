# LSTM log language model

from abc import abstractmethod
from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from log_analyzer.application import Application
from log_analyzer.config.config import Config
from log_analyzer.config.model_config import LSTMConfig, TieredLSTMConfig
from log_analyzer.model.attention import SelfAttention
from log_analyzer.model.model_util import initialize_weights


class LogModel(nn.Module):
    """Superclass for all log-data language models.

    All log-data language models should implement the forward() function with:

    input: input_sequence, lengths=None, mask=None, targets=None
    output: output_sequence, model_info?, loss=None

    Where model_info is only returned if the model is used internally by a tiered-model.
    model_info is any extra output produced by the model that's used by the tiered model (e.g. hidden states).
    model_info should either be a singular value or a tuple of values.
    Loss should be returned if targets is provided, otherwise None is returned.
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config: Config = config
        self.criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
        # Parameter setting
        self.tiered: bool = False

    def compute_loss(self, output: Tensor, Y: Tensor):
        """Computes the loss for the given model output and ground truth."""
        token_losses = self.criterion(output.transpose(1, 2), Y)
        line_losses = torch.mean(token_losses, dim=1)
        loss = torch.mean(line_losses, dim=0)

        # Return the loss, as well as extra details like loss per line
        return loss, line_losses.unsqueeze(-1), token_losses

    @abstractmethod
    def forward(self, sequences, lengths: Tensor = None, mask=None, targets=None):
        ...


class TieredLogModel(LogModel):
    """Superclass for tiered log-data language models."""

    def __init__(self, config: Config):
        super().__init__(config)
        # Parameter setting
        self.tiered: bool = True

    def compute_loss(self, output: Tensor, Y: Tensor):
        """Computes the loss for the given model output and ground truth."""
        loss = torch.tensor(0.0)
        line_losses_list = torch.empty(output.shape[:-2], dtype=torch.float)
        token_losses_list = torch.empty(output.shape[:-1], dtype=torch.float)
        if Application.instance().using_cuda:
            line_losses_list = line_losses_list.cuda()
        # output (num_steps x batch x length x embedding dimension)  Y
        # (num_steps x batch x length)
        for i, (step_output, step_y) in enumerate(zip(output, Y)):
            # On notebook, I checked it with forward LSTM and word
            # tokenization. Further checks have to be done...
            token_losses = self.criterion(step_output.transpose(1, 2), step_y)
            token_losses_list[i] = token_losses
            line_losses = torch.mean(token_losses, dim=1)
            line_losses_list[i] = line_losses
            step_loss = torch.mean(line_losses, dim=0)
            loss += step_loss.cpu()
        loss /= len(Y)
        return loss, line_losses_list, token_losses_list

    @abstractmethod
    def forward(self, sequences, lengths: Tensor = None, mask=None, targets=None):
        ...


class MultilineLogModel(LogModel):
    """Superclass for Multiline language models ("sentence-embedding"
    models)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.criterion = nn.MSELoss(reduction="none")
        # self.criterion = nn.CosineEmbeddingLoss(reduction="none")

    @abstractmethod
    def word_embedding(self, src):
        ...

    @abstractmethod
    def sentence_embedding(self, src):
        ...

    @abstractmethod
    def sentence_deembedding(self, src):
        ...

    def compute_loss(self, output: Tensor, Y: Tensor):
        """Computes the loss for the given model output and ground truth."""
        # If the shapes don't match the output was created via one-way sentence embedding and we need to do the same
        # Embedding on Y to compute loss.
        # If the shapes do match (but Output is a probability distribution over the vocabulary) we prefer to compute
        # loss in the vocabulary space (not sentence space)
        original_shape = Y.shape
        if output.shape[:3] != Y.shape:
            Y = self.word_embedding(Y)
            Y = self.sentence_embedding(Y)
        assert output.shape[:3] == Y.shape, f"Cannot reconcile output shape {output.shape} with target shape {Y.shape}"
        if isinstance(self.criterion, nn.CosineEmbeddingLoss):
            criterion_output = output.view(-1, output.shape[2])
            criterion_Y = Y.view(-1, Y.shape[2])
            targets = torch.ones((Y.shape[0] * Y.shape[1])).to(output.device)
            embedding_losses = self.criterion(criterion_output, criterion_Y, targets)
            embedding_losses = embedding_losses.view(Y.shape[0], Y.shape[1])
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            # Flatten dims 1 and 2 (line sequence, word) then transpose to: (batch, vocab_dim, sequence+word position)
            output = output.flatten(start_dim=1, end_dim=2).transpose(1, 2)
            criterion_Y = Y.flatten(start_dim=1, end_dim=2)
            embedding_losses = self.criterion(output, criterion_Y)
            # Reshape the loss tensor to (batch, line sequence, word)
            embedding_losses = embedding_losses.view(original_shape)
        else:
            embedding_losses = self.criterion(output, Y)
        line_losses = torch.mean(embedding_losses, dim=2) if len(embedding_losses.shape) > 2 else embedding_losses
        loss = torch.mean(line_losses[torch.all(Y, dim=2)])  # do not include loss from padding

        # Return the loss, as well as extra details like loss per line and per token
        return loss, line_losses, embedding_losses


class LSTMLanguageModel(LogModel):
    """Superclass for non-tiered LSTM log-data language models."""

    def __init__(self, config: LSTMConfig):
        super().__init__(config)

        self.config = config
        self.input_dim = config.input_dim

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
        seq_len: Optional[int]
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

        self.get_token_output = nn.Linear(fc_input_dim, config.vocab_size)

        # Weight initialization
        initialize_weights(self)

    @property
    def bidirectional(self):
        raise NotImplementedError("Bidirectional property has to be set in child class.")

    def forward(self, sequences, lengths: Tensor = None, mask=None, targets=None):
        """Performs token embedding, context-prepending if model is tiered, and
        runs the LSTM on the input sequences."""
        # batch size, sequence length, embedded dimension
        if self.tiered:
            lstm_in = sequences
        else:
            lstm_in = self.embeddings(sequences)

        if lengths is not None:
            if len(lengths.shape) > 1:
                lengths = lengths.squeeze(1)
            lstm_in = pack_padded_sequence(lstm_in, lengths.cpu(), enforce_sorted=False, batch_first=True)

        lstm_out, (hx, _) = self.stacked_lstm(lstm_in)

        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        return lstm_out, hx


class FwdLSTM(LSTMLanguageModel):
    """Standard forward LSTM model."""

    def __init__(self, config: LSTMConfig):
        self.name = "LSTM"
        super().__init__(config)

    def forward(self, sequences, lengths=None, mask=None, targets=None):
        """Handles attention (if relevant) and grabs the final token output
        guesses.

        Returns: predicted_tokens, (lstm_output_features, final_hidden_state), loss
        If targets is None then loss is returned as None
        """
        lstm_out, hx = super().forward(sequences, lengths)

        if self.has_attention:
            attention, _ = self.attention(lstm_out, mask)
            output = torch.cat((lstm_out, attention), dim=-1)
        else:
            output = lstm_out

        token_output = self.get_token_output(output)

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, *_ = self.compute_loss(token_output, targets)

        if self.tiered:
            return token_output, (lstm_out, hx), loss
        return token_output, loss

    @property
    def bidirectional(self):
        return False


class BidLSTM(LSTMLanguageModel):
    """Standard bidirectional LSTM model."""

    def __init__(self, config: LSTMConfig):
        self.name = "LSTM-Bid"
        super().__init__(config)

    def forward(self, sequences: torch.Tensor, lengths=None, mask=None, targets=None):
        """Handles bidir-state-alignment, attention (if relevant) and grabs the
        final token output guesses.

        Returns: predicted_tokens, (lstm_output_features, final_hidden_state), loss
        If targets is None then loss is returned as None
        """
        lstm_out, hx = super().forward(sequences, lengths)
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
            b_f_concat = torch.cat((b_f_concat, attention.squeeze(1)), dim=-1)

        token_output = self.get_token_output(b_f_concat)

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, *_ = self.compute_loss(token_output, targets)

        if self.tiered:
            return token_output, (lstm_out, hx), loss
        return token_output, loss

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

    def forward(self, context_input, context_prior_state: Tuple[Tensor, Tensor]):
        """Handles processing and updating of context info.

        Returns: context_output, (final_hidden_state, final_cell_state)
        """
        context_prior_hidden_state, context_prior_cell_state = context_prior_state
        output, (context_hidden_state, context_cell_state) = self.context_lstm_layers(
            context_input, (context_prior_hidden_state, context_prior_cell_state)
        )
        return output, (context_hidden_state, context_cell_state)


class TieredLSTM(TieredLogModel):
    """Tiered-LSTM model, combines a standard forward or bidirectional LSTM
    model for log-level analysis and a context LSTM for propagation of high-
    level context information."""

    config: TieredLSTMConfig

    def __init__(self, config: TieredLSTMConfig, bidirectional):

        super().__init__(config)
        # Parameter setting
        self.model: Type[LSTMLanguageModel]
        if bidirectional:
            self.model = BidLSTM
        else:
            self.model = FwdLSTM
        self.bidirectional = bidirectional

        event_model_layers = config.layers

        # User model state
        self.context_layers = (
            config.context_layers if isinstance(config.context_layers, list) else [config.context_layers]
        )
        self.context_lstm_input = None
        self.saved_lstm: Dict[int, Tuple[Tensor, Tensor, Tensor]] = {}

        # Layers
        self.event_level_lstm = self.model(config)
        self.event_level_lstm.tiered = True
        if self.bidirectional:
            self.context_input_features = event_model_layers[-1] * 4
        else:
            self.context_input_features = event_model_layers[-1] * 2
        self.context_lstm = ContextLSTM(config.context_layers, self.context_input_features)

        self.using_cuda = Application.instance().using_cuda

        # Weight initialization
        initialize_weights(self)

    def forward(self, sequences, lengths: Tensor = None, mask=None, targets=None):
        """Forward pass of tiered LSTM model.

        1. Applies context LSTM to the saved context information to generate context input for event LSTM.
        2. Applies event LSTM on the user sequences + context information
        3. Saves new context information for next forward pass

        Returns: predicted_tokens, loss
        If targets is None then loss is returned as None
        """
        # Split the input into users list and user sequences
        users, user_sequences = sequences
        # Convert users list to python list
        users = [user.item() for user in users]
        # Add a state list for any users we haven't seen before
        for user in users:
            if user not in self.saved_lstm:
                self.prepare_state(user)

        # Get the saved state for the users in the batch
        context_lstm_input, context_hidden_state, context_cell_state = self.get_batch_data(users)

        if self.event_level_lstm.bidirectional:
            token_output = torch.empty(
                (
                    user_sequences.shape[0],
                    user_sequences.shape[1],
                    user_sequences.shape[2] - 2,
                ),
                dtype=torch.float,
            )
        else:
            token_output = torch.empty_like(user_sequences, dtype=torch.float)

        token_output = token_output.unsqueeze(3).repeat(1, 1, 1, self.config.vocab_size)

        if Application.instance().using_cuda:
            token_output = token_output.cuda()
        # number of steps (e.g., 3), number of users (e.g., 64), lengths of
        # sequences (e.g., 10)
        for idx, sequence in enumerate(user_sequences):
            length = None if lengths is None else lengths[idx]
            if self.using_cuda and length is not None:
                length = length.cuda()
            # Apply the context LSTM to get context vector
            context_vector, (context_hidden_state, context_cell_state) = self.context_lstm(
                context_lstm_input,
                (context_hidden_state, context_cell_state),
            )

            x_lookups = self.event_level_lstm.embeddings(sequence)
            seq_length = sequence.shape[1]
            context = context_vector.tile([1, seq_length, 1])
            sequence = torch.cat([x_lookups, context], dim=-1)

            # Apply the event model to get token predictions
            event_model_output, (all_hidden, final_hidden), _ = self.event_level_lstm(sequence, lengths=length)
            if self.event_level_lstm.bidirectional:
                final_hidden = final_hidden.view(1, final_hidden.shape[1], -1)
            token_output[idx][
                : event_model_output.shape[0], : event_model_output.shape[1], : event_model_output.shape[2]
            ] = event_model_output

            # Save the mean hidden + final hidden for the event model
            if length is not None:
                mean_hidden = torch.sum(all_hidden, dim=1) / length.view(-1, 1)
            else:
                mean_hidden = torch.mean(all_hidden, dim=1)
            mean_final_hidden = torch.cat((mean_hidden, final_hidden[-1]), dim=1)
            context_lstm_input = torch.unsqueeze(mean_final_hidden, dim=1)

        # Update state for each user
        self.update_state(users, context_lstm_input, context_hidden_state, context_cell_state)

        loss = None
        if targets is not None:
            # Compute and return loss if targets is given
            loss, *_ = self.compute_loss(token_output, targets)

        return token_output, loss

    def prepare_state(self, user):
        """Set up the model state necessary to maintain user-specific state for
        this user.

        saved_lstm is a dict holding the tuple: (context_lstm_input,
        hidden_state, cell_state) for each user.
        """
        self.saved_lstm[user] = (
            torch.zeros((len(self.context_layers), self.context_input_features)),
            torch.zeros((len(self.context_layers), self.context_layers[0])),
            torch.zeros((len(self.context_layers), self.context_layers[0])),
        )
        if self.using_cuda:
            self.saved_lstm[user] = (
                self.saved_lstm[user][0].cuda(),
                self.saved_lstm[user][1].cuda(),
                self.saved_lstm[user][2].cuda(),
            )

    def get_batch_data(self, users):
        """Given a list of users, fetch the relevant history and model data for
        each user."""
        context_lstm_inputs = torch.tensor([])
        hidden_states = torch.tensor([])
        cell_states = torch.tensor([])
        if self.using_cuda:
            context_lstm_inputs = context_lstm_inputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
        # Loop over users
        for user in users:
            # Grab the context information
            context_lstm_inputs = torch.cat(
                (context_lstm_inputs, torch.unsqueeze(self.saved_lstm[user][0].detach(), dim=0)), dim=0
            )
            hidden_states = torch.cat((hidden_states, torch.unsqueeze(self.saved_lstm[user][1].detach(), dim=0)), dim=0)
            cell_states = torch.cat((cell_states, torch.unsqueeze(self.saved_lstm[user][2].detach(), dim=0)), dim=0)
        # Transpose the h and c states to order (num_steps, batchsize, sequence)
        hidden_states = torch.transpose(hidden_states, 0, 1)
        cell_states = torch.transpose(cell_states, 0, 1)
        return context_lstm_inputs, hidden_states, cell_states

    def update_state(self, users, context_lstm_inputs, hidden_states, cell_states):
        """Given one batch of history/model data output by the model, update
        the stored state for future use."""
        # Transpose the h and c states to order (batchsize, num_steps, sequence)
        hidden_states = torch.transpose(hidden_states, 0, 1)
        cell_states = torch.transpose(cell_states, 0, 1)
        for user, context_lstm_input, hidden_state, cell_state in zip(
            users, context_lstm_inputs, hidden_states, cell_states
        ):
            self.saved_lstm[user] = (context_lstm_input, hidden_state, cell_state)
