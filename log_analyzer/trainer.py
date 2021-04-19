import torch
from torch.functional import split
import torch.nn as nn
from log_analyzer.model.lstm import Fwd_LSTM, Bid_LSTM, Tiered_LSTM
import log_analyzer.model.auxiliary as auxiliary


class Trainer():  # TODO name this something more descriptive, it might be used as a wrapper around both transformer/LSTM

    def __init__(self, args, conf, checkpoint_dir, data_handler = None, lr=1e-3, step_size=20, gamma=0.99, patience=20, verbose=False):

        # Check GPU
        self.cuda = torch.cuda.is_available()

        self.jagged = args.jagged
        self.tiered = args.tiered
        self.data_handler = data_handler

        # Select a model
        if args.tiered:
            self.model =  Tiered_LSTM(args.lstm_layers, args.context_layers, conf['token_set_size'], args.embed_dim, jagged = args.jagged, bid = args.bidirectional)
        else:
            if args.bidirectional:
                model = Bid_LSTM
            else:
                model = Fwd_LSTM

            # Create a model
            self.model = model(
                args.lstm_layers, conf['token_set_size'], args.embed_dim, jagged=args.jagged)
        if self.cuda:
            self.model.cuda()

        # Create settings for training.
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        self.early_stopping = auxiliary.EarlyStopping(
            patience=patience, verbose=verbose, path=checkpoint_dir)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

    def compute_loss(self, X, Y, lengths, mask, ctxt_vector, ctxt_hidden, ctxt_cell):
        """Computes the loss for the given input."""
        if self.tiered: # For tiered model. 
            loss = 0
            output, ctxt_vector, ctxt_h, ctxt_c = self.model(X, ctxt_vector, ctxt_hidden, ctxt_cell, lengths = lengths)    
            self.data_handler.update_state(ctxt_vector, ctxt_h, ctxt_c)
            for step_output, true_y in zip(output, Y): # output (num_steps x batch x length x embedding dimension)  Y (num_steps x batch x length)
                token_losses = self.criterion(step_output.transpose(1,2), true_y)
                if self.jagged: # On notebook, I checked it with forward LSTM and word tokenization. Further checks have to be done... 
                    token_losses = self.criterion(output.transpose(1, 2), Y[:,:max(lengths)])
                    masked_losses = token_losses * M
                    line_losses = torch.sum(masked_losses, dim = 1)
                else:
                    token_losses = self.criterion(output.transpose(1, 2), Y)
                    line_losses = torch.mean(token_losses, dim = 1)
                step_loss = torch.mean(line_losses, dim = 0)
                loss += step_loss
        else: # For non-tiered models.
            output, lstm_out, hx = self.model(X, lengths=lengths)
            if self.jagged:
                token_losses = self.criterion(output.transpose(1, 2), Y[:,:max(lengths)])
                masked_losses = token_losses * mask[:,:max(lengths)]
                line_losses = torch.sum(masked_losses, dim=1)
            else:
                token_losses = self.criterion(output.transpose(1, 2), Y)
                line_losses = torch.mean(token_losses, dim=1)
            loss = torch.mean(line_losses, dim=0)

        return loss

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
        if self.tiered:
            C_V =  batch['context_vector']
            C_H = batch['c_state_init']
            C_C = batch['h_state_init']
        else:
            C_V = None
            C_H = None
            C_C = None
        if self.cuda:
            X = X.cuda()
            Y = Y.cuda()
            if self.jagged:
                L = L.cuda()
                M = M.cuda()
            if self.tiered:
                C_V = C_V.cuda()
                C_H = C_H.cuda()
                C_C = C_C.cuda()

        return X, Y, L, M, C_V, C_H, C_C

    def train_step(self, batch):
        """Defines a single training step. Feeds data through the model, computes the loss and makes an optimization step."""

        self.model.train()
        self.optimizer.zero_grad()

        X, Y, L, M, C_V, C_H, C_C = self.split_batch(batch)

        loss = self.compute_loss(X, Y, lengths=L, mask=M, ctxt_vector = C_V, ctxt_hidden = C_H, ctxt_cell = C_C)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.early_stopping(loss, self.model)

        return loss, self.early_stopping.early_stop

    def eval_step(self, batch):
        """Defines a single evaluation step. Feeds data through the model and computes the loss."""
        # TODO add more metrics, like perplexity.
        self.model.eval()

        X, Y, L, M, C_V, C_H, C_C = self.split_batch(batch)

        token_losses = self.compute_loss(X, Y, lengths=L, mask=M, ctxt_vector = C_V, ctxt_hidden = C_H, ctxt_cell = C_C)

        return token_losses
