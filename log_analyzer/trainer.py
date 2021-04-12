import torch
import torch.nn as nn
from log_analyzer.model.lstm import Fwd_LSTM, Bid_LSTM
import log_analyzer.model.auxiliary as auxiliary

def training_settings(args, conf, lr = 1e-3, step_size = 20, gamma = 0.99, patience = 20, verbose = False):


    # Check GPU
    cuda = torch.cuda.is_available()
    
    # Select a model
    if args.bidirectional:
        model = Bid_LSTM
    else:
        model = Fwd_LSTM

    # Create a model
    model = model(args.lstm_layers, conf['token_set_size'], args.embed_dim, jagged=args.jagged)
    if cuda:
        model.cuda()

    # Create settings for training.
    criterion = nn.CrossEntropyLoss(reduction= 'none')
    early_stopping = auxiliary.EarlyStopping(patience=patience, verbose=verbose)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= step_size, gamma=gamma)

    return model, criterion,  optimizer, scheduler, early_stopping, cuda

def train_model(batch, model, criterion, optimizer, scheduler, early_stopping, cuda, jagged):

    model.train()
    optimizer.zero_grad()
    X = batch['x']
    Y = batch['t']
    if jagged:
        L = batch['length']
        M = batch['mask']
    else:
        L = None
        M = None
    if cuda:
        X = X.cuda()
        Y = Y.cuda()
        if jagged:
            L = L.cuda()
            M = M.cuda()
    output, lstm_out, hx = model(X, lengths = L) 
    token_losses = criterion(output.transpose(1,2), Y)
    if jagged:
        masked_losses = token_losses * M
        line_losses = torch.sum(masked_losses, dim = 1)
    else:
        line_losses = torch.mean(token_losses, dim = 1)
        
    loss = torch.mean(line_losses, dim = 0)

    loss.backward()
    optimizer.step()
    scheduler.step()

    early_stopping(loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        early_stopping.early_stop = False
        early_stopping.counter = 0
    
    return model