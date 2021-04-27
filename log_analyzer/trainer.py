import torch
import torch.nn as nn
from log_analyzer.model.lstm import Fwd_LSTM, Bid_LSTM
import log_analyzer.model.auxiliary as auxiliary

# TODO name this something more descriptive, it might be used as a wrapper around both transformer/LSTM
class Trainer():

    @property
    def model(self):
        raise NotImplementedError(
            "Model type to be overriddden in child class.")


    def __init__(self, args, conf, checkpoint_dir, data_handler=None, verbose=False):

        # Check GPU
        self.cuda = torch.cuda.is_available()

        self.jagged = args.jagged
        self.bidirectional = args.bidirectional
        self.data_handler = data_handler

        if self.cuda:
            self.model.cuda()

        # Create settings for training.
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        self.early_stopping = auxiliary.EarlyStopping(
            patience=conf["patience"], verbose=verbose, path=checkpoint_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=conf['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=conf['step_size'], gamma=conf['gamma'])

    def compute_loss(self, X, Y, lengths, mask):
        """Computes the loss for the given input."""
        output, _, _ = self.model(X, lengths=lengths)
        if self.jagged:
            if self.bidirectional:
                token_losses = self.criterion(
                    output.transpose(1, 2), Y[:, 1:max(lengths-1)])
                masked_losses = token_losses * mask[:, 1:max(lengths-1)]
            else:
                token_losses = self.criterion(
                    output.transpose(1, 2), Y[:, :max(lengths)])
                masked_losses = token_losses * mask[:, :max(lengths)]
            line_losses = torch.sum(masked_losses, dim=1)
        else:
            token_losses = self.criterion(output.transpose(1, 2), Y)
            line_losses = torch.mean(token_losses, dim=1)
        loss = torch.mean(line_losses, dim=0)

        return loss

    def optimizer_step(self, loss):
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.early_stopping(loss, self.model)

    def split_batch(self, batch):
        """Splits a batch into variables containing relevant data."""
        X = batch['x']
        Y = batch['t']
        if self.jagged:
            L = batch['length']
            M = batch['mask']
        else:
            L = None
            M = None
        if self.cuda:
            X = X.cuda()
            Y = Y.cuda()
            if self.jagged:
                L = L.cuda()
                M = M.cuda()

        return X, Y, L, M

    def train_step(self, batch):
        """Defines a single training step. Feeds data through the model, computes the loss and makes an optimization step."""

        self.model.train()
        self.optimizer.zero_grad()

        X, Y, L, M = self.split_batch(batch)

        loss = self.compute_loss(
            X, Y, lengths=L, mask=M)

        self.optimizer_step(loss)

        return loss, self.early_stopping.early_stop

    def eval_step(self, batch):
        """Defines a single evaluation step. Feeds data through the model and computes the loss."""
        # TODO add more metrics, like perplexity.
        self.model.eval()

        X, Y, L, M = self.split_batch(batch)

        token_losses = self.compute_loss(
            X, Y, lengths=L, mask=M)

        return token_losses


class LSTMTrainer(Trainer):
    """Trainer class for forward and bidirectional LSTM model"""
    @property
    def model(self):
        if self.lstm is None:
            raise RuntimeError("Model not intialized!")
        return self.lstm

    def __init__(self, args, conf, checkpoint_dir, data_handler, verbose):

        if args.bidirectional:
            model = Bid_LSTM
        else:
            model = Fwd_LSTM
        # Create a model
        self.lstm = model(
            args.lstm_layers, conf['token_set_size'], args.embed_dim, jagged=args.jagged)

        super().__init__(args, conf, checkpoint_dir, data_handler=data_handler, verbose=verbose)
