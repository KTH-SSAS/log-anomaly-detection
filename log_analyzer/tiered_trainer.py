import torch
from log_analyzer.trainer import Trainer
from log_analyzer.model.lstm import Tiered_LSTM

class TieredTrainer(Trainer):
    """Trainer class for tiered LSTM model"""
    @property
    def model(self):
        if self.lstm is None:
            raise RuntimeError("Model not intialized!")
        return self.lstm

    def __init__(self, args, conf, checkpoint_dir, data_handler, verbose):

        self.lstm = Tiered_LSTM(args.lstm_layers, args.context_layers,
                                 conf['token_set_size'], args.embed_dim, jagged=args.jagged, bid=args.bidirectional)
        super().__init__(args, conf, checkpoint_dir, data_handler=data_handler, verbose=verbose)

    def compute_loss(self, X, Y, lengths, mask, ctxt_vector, ctxt_hidden, ctxt_cell):
        """Computes the loss for the given input."""

        loss = 0
        output, ctxt_vector, ctxt_h, ctxt_c = self.model(
            X, ctxt_vector, ctxt_hidden, ctxt_cell, lengths=lengths)
        self.data_handler.update_state(ctxt_vector, ctxt_h, ctxt_c)
        # output (num_steps x batch x length x embedding dimension)  Y (num_steps x batch x length)
        for i, (step_output, true_y) in enumerate(zip(output, Y)):
            if self.jagged:  # On notebook, I checked it with forward LSTM and word tokenization. Further checks have to be done...
                token_losses = self.criterion(
                    step_output.transpose(1, 2), true_y[:, :max(lengths[i])])
                masked_losses = token_losses * mask[i][:, :max(lengths[i])]
                line_losses = torch.sum(masked_losses, dim=1)
            else:
                token_losses = self.criterion(
                    step_output.transpose(1, 2), true_y)
                line_losses = torch.mean(token_losses, dim=1)
            step_loss = torch.mean(line_losses, dim=0)
            loss += step_loss
        loss /= len(X)
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

    def eval_step(self, batch):
        """Defines a single evaluation step. Feeds data through the model and computes the loss."""
        self.model.eval()

        X, Y, L, M, C_V, C_H, C_C = self.split_batch(batch)

        token_losses = self.compute_loss(
            X, Y, lengths=L, mask=M, ctxt_vector=C_V, ctxt_hidden=C_H, ctxt_cell=C_C)

        return token_losses

    def train_step(self, batch):
        """Defines a single training step. Feeds data through the model, computes the loss and makes an optimization step."""

        self.model.train()
        self.optimizer.zero_grad()

        X, Y, L, M, C_V, C_H, C_C = self.split_batch(batch)

        loss = self.compute_loss(
            X, Y, lengths=L, mask=M, ctxt_vector=C_V, ctxt_hidden=C_H, ctxt_cell=C_C)

        self.optimizer_step(loss)

        return loss, self.early_stopping.early_stop