import torch
from torch.functional import split
import torch.nn as nn
from log_analyzer.model.lstm import Fwd_LSTM, Bid_LSTM
import log_analyzer.model.auxiliary as auxiliary

class Trainer(): #TODO name this something more descriptive, it might be used as a wrapper around both transformer/LSTM

    def __init__(self, args, conf, lr = 1e-3, step_size = 20, gamma = 0.99, patience = 20, verbose = False):

        # Check GPU
        self.cuda = torch.cuda.is_available()

        self.jagged = args.jagged
        
        # Select a model
        if args.bidirectional:
            model = Bid_LSTM
        else:
            model = Fwd_LSTM

        # Create a model
        self.model = model(args.lstm_layers, conf['token_set_size'], args.embed_dim, jagged=args.jagged)
        if self.cuda:
            self.model.cuda()

        # Create settings for training.
        self.criterion = nn.CrossEntropyLoss(reduction= 'none')
        self.early_stopping = auxiliary.EarlyStopping(patience=patience, verbose=verbose)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= step_size, gamma=gamma)

    def compute_loss(self, X, Y, lengths, mask):
        """Computes the loss for the given input."""
        output, lstm_out, hx = self.model(X, lengths) 
        token_losses = self.criterion(output.transpose(1,2), Y)
        if self.jagged:
            masked_losses = token_losses * mask
            line_losses = torch.sum(masked_losses, dim = 1)
        else:
            line_losses = torch.mean(token_losses, dim = 1)
            
        loss = torch.mean(line_losses, dim = 0)
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

        loss = self.compute_loss(X, Y, lengths=L, mask=M)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.early_stopping(loss, self.model)

        if self.early_stopping.early_stop:
            print("Early stopping")
            self.early_stopping.early_stop = False
            self.early_stopping.counter = 0
        
        return loss
    
    def eval_step(self, batch): 
        """Defines a single evaluation step. Feeds data through the model and computes the loss."""
        # TODO add more metrics, like perplexity.
        self.model.eval()
        
        X, Y, L, M = self.split_batch(batch)
        
        token_losses = self.compute_loss(X, Y, lengths=L, mask=M)

        return token_losses
    