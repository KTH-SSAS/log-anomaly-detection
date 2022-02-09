import torch

from log_analyzer.config.model_config import ModelConfig, TieredLSTMConfig, TieredTransformerConfig
from log_analyzer.config.trainer_config import TrainerConfig
from log_analyzer.model.lstm import TieredLSTM
from log_analyzer.model.transformer import TieredTransformer
from log_analyzer.trainer import Trainer
from log_analyzer.data.data_loader import TieredTransformerBatcher


class TieredTrainer(Trainer):
    """Trainer class for tiered LSTM model."""

    def __init__(
        self,
        config: TrainerConfig,
        model_config: ModelConfig,
        bidirectional,
        checkpoint_dir,
        train_loader,
        test_loader,
    ):

        self.train_loader = train_loader
        self.test_loader = test_loader
        super().__init__(config, checkpoint_dir)

    def compute_loss(self, output, Y, lengths, mask):
        """Computes the loss for the given model output and ground truth."""
        loss = 0
        line_losses_list = torch.empty(output.shape[:-2], dtype=torch.float)
        if self.cuda:
            line_losses_list = line_losses_list.cuda()
        if lengths is not None:
            targets = Y[:, :, : torch.max(lengths)]
        else:
            targets = Y
        # output (num_steps x batch x length x embedding dimension)  Y
        # (num_steps x batch x length)
        for i, (step_output, step_y) in enumerate(zip(output, Y)):
            # On notebook, I checked it with forward LSTM and word
            # tokenization. Further checks have to be done...
            if lengths is not None:
                token_losses = self.criterion(step_output.transpose(1, 2), step_y[:, : torch.max(lengths)])
                masked_losses = token_losses * mask[i][:, : torch.max(lengths)]
                line_losses = torch.sum(masked_losses, dim=1)
            else:
                token_losses = self.criterion(step_output.transpose(1, 2), step_y)
                line_losses = torch.mean(token_losses, dim=1)
            line_losses_list[i] = line_losses
            step_loss = torch.mean(line_losses, dim=0)
            loss += step_loss
        loss /= len(Y)
        return loss, line_losses_list, targets

    def train_step(self, batch):
        """Defines a single training step.

        Feeds data through the model, computes the loss and makes an
        optimization step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Split the batch into input, ground truth, etc.
        X, Y, L, M, model_info = self.split_batch(batch)

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Apply the model to input to produce the output
                output = self.run_model(X, L, model_info, self.train_loader)

                # Compute the loss for the output
                loss, *_ = self.compute_loss(output, Y, lengths=L, mask=M)
        else:
            # Apply the model to input to produce the output
            output = self.run_model(X, L, model_info, self.train_loader)

            # Compute the loss for the output
            loss, *_ = self.compute_loss(output, Y, lengths=L, mask=M)

        # Take an optimization step based on the loss
        self.optimizer_step(loss)

        return loss, self._EarlyStopping.early_stop

    def eval_step(self, batch, store_eval_data=False):
        """Defines a single evaluation step.

        Feeds data through the model and computes the loss.
        """
        self.model.eval()

        output, Y, L, M = self.eval_model(batch, self.test_loader)

        # Compute the loss for the output
        loss, line_losses, targets = self.compute_loss(output, Y, lengths=L, mask=M)

        # Save the results if desired
        if store_eval_data:
            preds = torch.argmax(output, dim=-1)
            self.evaluator.add_evaluation_data(
                torch.flatten(targets, end_dim=1),
                torch.flatten(preds, end_dim=1),
                torch.flatten(batch["user"], end_dim=1),
                torch.flatten(line_losses, end_dim=1),
                torch.flatten(batch["second"], end_dim=1),
                torch.flatten(batch["red"], end_dim=1),
            )

        return loss, output

    def run_model(batch):
        pass

    def eval_model(batch):
        pass


class TieredLSTMTrainer(TieredTrainer):
    @property
    def model(self):
        if self.lstm is None:
            raise RuntimeError("Model not intialized!")
        return self.lstm

    def __init__(
        self,
        config: TrainerConfig,
        lstm_config: TieredLSTMConfig,
        bidirectional,
        checkpoint_dir,
        train_loader,
        test_loader,
    ):

        self.lstm = TieredLSTM(lstm_config, bidirectional)
        super().__init__(config, lstm_config, bidirectional, checkpoint_dir, train_loader, test_loader)

    def split_batch(self, batch):
        """Splits a batch into variables containing relevant data."""

        X, Y, L, M = super().split_batch(batch)

        C_V = batch["context_vector"]
        C_H = batch["c_state_init"]
        C_C = batch["h_state_init"]

        if self.cuda:
            C_V = C_V.cuda()
            C_H = C_H.cuda()
            C_C = C_C.cuda()

        return X, Y, L, M, (C_V, C_H, C_C)

    def run_model(self, X, L, model_info, data_loader):
        ctxt_vector = model_info[0]
        ctxt_hidden = model_info[1]
        ctxt_cell = model_info[2]

        output, ctxt_vector, ctxt_hidden, ctxt_cell = self.model(X, ctxt_vector, ctxt_hidden, ctxt_cell, lengths=L)
        data_loader.update_state(ctxt_vector, ctxt_hidden, ctxt_cell)

        return output

    def eval_model(self, batch, data_loader):

        # Split the batch into input, ground truth, etc.
        X, Y, L, M, model_info = self.split_batch(batch)

        # Apply the model to input to produce the output
        output = self.run_model(X, L, model_info, data_loader)

        return output, Y, L, M


class TieredTransformerTrainer(TieredTrainer):
    """Trainer class for Transformer model."""

    @property
    def model(self):
        if self.transformer is None:
            raise RuntimeError("Model not initialized!")
        return self.transformer

    def __init__(
        self,
        config: TrainerConfig,
        transformer_config: TieredTransformerConfig,
        bidirectional,
        checkpoint_dir,
        train_loader,
        test_loader,
    ):
        # Create a model
        self.transformer = TieredTransformer(transformer_config)
        super().__init__(config, transformer_config, bidirectional, checkpoint_dir, train_loader, test_loader)

    def split_batch(self, batch):
        """Splits a batch into variables containing relevant data."""

        X, Y, L, M = super().split_batch(batch)

        # C_V = batch["context_vector"]
        C_H = batch["history"]
        H_L = batch["history_length"]

        if self.cuda:
            # C_V = C_V.cuda()
            C_H = C_H.cuda()

        return X, Y, L, M, (C_H, H_L)

    def run_model(self, X, L, model_info, data_loader: TieredTransformerBatcher):
        history = model_info[0]
        history_length = model_info[1]

        output, ctxt_vector, history = self.model(X, history, lengths=L)
        data_loader.update_state(history)

        return output

    def eval_model(self, batch, data_loader):

        # Split the batch into input, ground truth, etc.
        X, Y, L, M, model_info = self.split_batch(batch)

        output = self.run_model(X, L, model_info, data_loader)

        return output, Y, L, M
