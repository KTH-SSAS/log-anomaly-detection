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

    def train_step(self, split_batch):
        """Defines a single training step.

        Feeds data through the model, computes the loss and makes an
        optimization step.
        """
        X = split_batch["X"]
        Y = split_batch["Y"]
        L = split_batch["L"]
        M = split_batch["M"]
        model_info = split_batch["model_info"]

        self.model.train()
        self.optimizer.zero_grad()

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Apply the model to input to produce the output
                output, model_info, _ = self.model(X, model_info, lengths=L, mask=M)
                # Update the dataloader's state with new model_info (context)
                self.train_loader.update_state(model_info)

                # Compute the loss for the output
                loss, _ = self.model.compute_loss(output, Y, lengths=L, mask=M)
        else:
            # Apply the model to input to produce the output
            output, model_info, _ = self.model(X, model_info, lengths=L, mask=M)
            # Update the dataloader's state with new model_info (context)
            self.train_loader.update_state(model_info)

            # Compute the loss for the output
            loss, _ = self.model.compute_loss(output, Y, lengths=L, mask=M)

        # Take an optimization step based on the loss
        self.optimizer_step(loss)

        return loss, self._EarlyStopping.early_stop

    def eval_step(self, split_batch, store_eval_data=False):
        """Defines a single evaluation step.

        Feeds data through the model and computes the loss.
        """
        X = split_batch["X"]
        Y = split_batch["Y"]
        L = split_batch["L"]
        M = split_batch["M"]
        model_info = split_batch["model_info"]

        users = split_batch["user"]
        seconds = split_batch["second"]
        red_flags = split_batch["red_flag"]

        self.model.eval()

        # Apply the model to input to produce the output
        output, model_info, _ = self.model(X, model_info, lengths=L)
        # Update the dataloader's state with new model_info (context)
        self.test_loader.update_state(model_info)

        if L is not None:
            max_length = int(torch.max(L))
            Y = Y[:, :, :max_length]

        # Compute the loss for the output
        loss, line_losses = self.model.compute_loss(output, Y, lengths=L, mask=M)

        # Save the results if desired
        if store_eval_data:
            preds = torch.argmax(output, dim=-1)
            self.evaluator.add_evaluation_data(
                torch.flatten(Y, end_dim=1),
                torch.flatten(preds, end_dim=1),
                torch.flatten(users, end_dim=1),
                torch.flatten(line_losses, end_dim=1),
                torch.flatten(seconds, end_dim=1),
                torch.flatten(red_flags, end_dim=1),
            )

        return loss, output


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
        # Create the model
        self.lstm = TieredLSTM(lstm_config, bidirectional)
        super().__init__(config, lstm_config, bidirectional, checkpoint_dir, train_loader, test_loader)


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
        # Create the model
        self.transformer = TieredTransformer(transformer_config)
        super().__init__(config, transformer_config, bidirectional, checkpoint_dir, train_loader, test_loader)
