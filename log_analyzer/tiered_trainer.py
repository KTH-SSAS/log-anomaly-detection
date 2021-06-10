from log_analyzer.config.trainer_config import TrainerConfig
from log_analyzer.config.model_config import TieredLSTMConfig
import torch
from log_analyzer.trainer import Trainer
from log_analyzer.model.lstm import Tiered_LSTM
from log_analyzer.data.data_loader import OnlineLMBatcher


class TieredTrainer(Trainer):
    """Trainer class for tiered LSTM model"""
    @property
    def model(self):
        if self.lstm is None:
            raise RuntimeError("Model not intialized!")
        return self.lstm

    def __init__(self, config: TrainerConfig, lstm_config: TieredLSTMConfig, checkpoint_dir, verbose, data_handler):

        self.lstm = Tiered_LSTM(lstm_config)
        self.data_handler = data_handler
        super().__init__(config, verbose, checkpoint_dir)

    def compute_loss(self, output, Y, lengths, mask):
        """Computes the loss for the given model output and ground truth."""
        loss = 0
        # output (num_steps x batch x length x embedding dimension)  Y (num_steps x batch x length)
        for i, (step_output, true_y) in enumerate(zip(output, Y)):
            if self.jagged:  # On notebook, I checked it with forward LSTM and word tokenization. Further checks have to be done...
                skip_len = 2 if self.bidirectional else 0
                token_losses = self.criterion(
                    step_output.transpose(1, 2), true_y[:, :max(lengths)-skip_len])
                masked_losses = token_losses * mask[i][:max(lengths-skip_len)]
                line_losses = torch.sum(masked_losses, dim=1)
            else:
                token_losses = self.criterion(
                    step_output.transpose(1, 2), true_y)
                line_losses = torch.mean(token_losses, dim=1)
            step_loss = torch.mean(line_losses, dim=0)
            loss += step_loss
        loss /= len(Y)
        return loss

    def split_batch(self, batch):
        """Splits a batch into variables containing relevant data."""

        X, Y, L, M = super().split_batch(batch)

        C_V = batch['context_vector']
        C_H = batch['c_state_init']
        C_C = batch['h_state_init']

        if self.cuda:
            C_V = C_V.cuda()
            C_H = C_H.cuda()
            C_C = C_C.cuda()

        return X, Y, L, M, C_V, C_H, C_C

    def train_step(self, batch):
        """Defines a single training step. Feeds data through the model, computes the loss and makes an optimization step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Split the batch into input, ground truth, etc.
        X, Y, L, M, ctxt_vector, ctxt_hidden, ctxt_cell = self.split_batch(
            batch)

        # Apply the model to input to produce the output
        output, ctxt_vector, ctxt_hidden, ctxt_cell = self.model(
            X, ctxt_vector, ctxt_hidden, ctxt_cell, lengths=L)
        self.data_handler.update_state(ctxt_vector, ctxt_hidden, ctxt_cell)

        # Compute the loss for the output
        loss = self.compute_loss(
            output, Y, lengths=L, mask=M)

        # Take an optimization step based on the loss
        self.optimizer_step(loss)

        return loss, self.early_stopping.early_stop

    def eval_step(self, batch):
        """Defines a single evaluation step. Feeds data through the model and computes the loss."""
        self.model.eval()

        # Split the batch into input, ground truth, etc.
        X, Y, L, M, ctxt_vector, ctxt_hidden, ctxt_cell = self.split_batch(
            batch)

        # Apply the model to input to produce the output
        output, ctxt_vector, ctxt_hidden, ctxt_cell = self.model(
            X, ctxt_vector, ctxt_hidden, ctxt_cell, lengths=L)
        self.data_handler.update_state(ctxt_vector, ctxt_hidden, ctxt_cell)

        # Compute the loss for the output
        loss = self.compute_loss(
            output, Y, lengths=L, mask=M)

        return loss, output
