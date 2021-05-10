from argparse import ArgumentParser
import os
import json
import socket
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import log_analyzer.data.data_loader as data_utils
from log_analyzer.trainer import LSTMTrainer
from log_analyzer.tiered_trainer import TieredTrainer
try:
    import torch
except ImportError:
    print('PyTorch is needed for this application.')

"""
Helper functions for model creation and training
"""

def create_identifier_string(model_name, comment=""):
    #TODO have model name be set by config, args or something else
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    id = f'{model_name}_{current_time}_{socket.gethostname()}_{comment}'
    return id


def create_model(args):
    """Creates a model plus trainer given the specifications in args"""
    base_logdir = './runs/'
    id_string = create_identifier_string("lstm")
    log_dir = os.path.join(base_logdir, id_string)
    os.mkdir(log_dir)

    # Read a config file.   
    with open(args.config, 'r') as f:
        conf = json.load(f)

    # Settings for dataloader.
    sentence_length = conf["sentence_length"] - 1 - int(args.skipsos) + int(args.bidirectional)
    train_days = conf['train_files']
    test_days = conf['test_files']
    train_loader, test_loader = data_utils.load_data(train_days, test_days, args, sentence_length)

    # Settings for LSTM.
    if args.tiered:
        trainer_class = TieredTrainer
    else:
        trainer_class = LSTMTrainer
    
    lm_trainer = trainer_class(args, conf, log_dir, verbose = True, data_handler = train_loader)

    return lm_trainer


def train_model(args, lm_trainer):
    """Perform 1 epoch of training on lm_trainer"""

    # Read a config file.   
    with open(args.config, 'r') as f:
        conf = json.load(f)

    # Settings for dataloader.
    sentence_length = conf["sentence_length"] - 1 - int(args.skipsos) + int(args.bidirectional)
    train_days = conf['train_files']
    test_days = conf['test_files']
    train_loader, test_loader = data_utils.load_data(train_days, test_days, args, sentence_length)

    outfile = None
    verbose = False
    log_dir = lm_trainer.checkpoint_dir
    writer = SummaryWriter(os.path.join(log_dir, 'metrics'))

    train_losses = []
    for iteration, batch in enumerate(train_loader):
        if args.tiered:
            if train_loader.flush is False:
                loss, done= lm_trainer.train_step(batch)
            else:
                loss, *_ = lm_trainer.eval_step(batch)
                print(f'Due to flush, training stopped... Current loss: {loss:.3f}')
        else: 
            loss, done= lm_trainer.train_step(batch)
        train_losses.append(loss.item())
        writer.add_scalar(f'Loss/train_day_{batch["day"][0]}', loss, iteration)
        if done:
            print("Early stopping.")
            break

    test_losses = []
    for iteration, batch in enumerate(test_loader):
        loss, *_ = lm_trainer.eval_step(batch)
        test_losses.append(loss.item())
        writer.add_scalar(f'Loss/test_day_{batch["day"][0]}', loss, iteration)

        if outfile is not None:
            for line, sec, day, usr, red, loss in zip(batch['line'].flatten().tolist(),
                                                    batch['second'].flatten().tolist(),
                                                    batch['day'].flatten().tolist(),
                                                    batch['user'].flatten().tolist(),
                                                    batch['red'].flatten().tolist(),
                                                    loss.flatten().tolist()):
                outfile.write('%s %s %s %s %s %s %r\n' % (iteration, line, sec, day, usr, red, loss))

        if verbose:
            print(f"{batch['x'].shape[0]}, {batch['line'][0]}, {batch['second'][0]} fixed {batch['day']} {loss}") 
            # TODO: I don't think this print line, but I decided to keep it since removing a line is always easier than adding a line.
            #       Also, In the original code, there was {data.index} which seems to be an accumulated sum of batch sizes. 
            #       I don't think we need {data.index}. but... I added it to to-do since we might need to do it in future.
    
    writer.close()
    torch.save(lm_trainer.model, os.path.join(log_dir,'model.pt'))
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(conf, f)
    return train_losses, test_losses
