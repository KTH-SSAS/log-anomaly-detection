import torch

from log_analyzer.config.model_config import ModelConfig, TieredLSTMConfig, TieredTransformerConfig
from log_analyzer.config.trainer_config import TrainerConfig
from log_analyzer.model.lstm import TieredLSTM
from log_analyzer.model.transformer import TieredTransformer
from log_analyzer.trainer import Trainer


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

    def train_step(self, X, Y, L, M, model_info):
        """Defines a single training step.

        Feeds data through the model, computes the loss and makes an
        optimization step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Apply the model to input to produce the output
                output = self.run_model(X, L, model_info, self.train_loader)

                # Compute the loss for the output
                loss, *_ = self.model.compute_loss(output, Y, lengths=L, mask=M)
        else:
            # Apply the model to input to produce the output
            output = self.run_model(X, L, model_info, self.train_loader)

            # Compute the loss for the output
            loss, *_ = self.model.compute_loss(output, Y, lengths=L, mask=M)

        # Take an optimization step based on the loss
        self.optimizer_step(loss)

        return loss, self._EarlyStopping.early_stop

    def eval_step(self, X, Y, L, M, model_info, store_eval_data=False, batch=None):
        """Defines a single evaluation step.

        Feeds data through the model and computes the loss.
        """
        self.model.eval()

        output = self.eval_model(X, L, model_info, self.test_loader)

        # Compute the loss for the output
        loss, line_losses, targets = self.model.compute_loss(output, Y, lengths=L, mask=M)

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

    def run_model(self, X, L, model_info, data_loader):
        ctxt_vector = model_info[0]
        ctxt_hidden = model_info[1]
        ctxt_cell = model_info[2]

        output, ctxt_vector, ctxt_hidden, ctxt_cell = self.model(X, ctxt_vector, ctxt_hidden, ctxt_cell, lengths=L)
        data_loader.update_state(ctxt_vector, ctxt_hidden, ctxt_cell)

        return output

    def eval_model(self, X, L, model_info, data_loader):
        # Apply the model to input to produce the output
        output = self.run_model(X, L, model_info, data_loader)
        return output


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

    def run_model(self, X, L, model_info, data_loader):
        ctxt_vector = model_info[0]
        history = model_info[1]
        history_length = model_info[2]

        output, ctxt_vector, history = self.model(X, ctxt_vector, history, lengths=L)
        data_loader.update_state(ctxt_vector, history)

        return output

    def eval_model(self, X, L, model_info, data_loader):
        output = self.run_model(X, L, model_info, data_loader)
        return output
