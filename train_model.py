from argparse import ArgumentParser
import argparse
import numpy as np
import os
import json
import log_analyzer.data.utils as data_utils
import log_analyzer.model.lstm as lstms
from log_analyzer.trainer import Trainer
from log_analyzer import evaluator
import torch
from torch.utils.tensorboard import SummaryWriter
import socket
from datetime import datetime

"""
Entrypoint script for training
"""

def create_identifier_string(model_name, comment=""):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    id = f'{model_name}_{current_time}_{socket.gethostname()}_{comment}'
    return id


def train(args):

    # Read a config file.   
    with open(args.config, 'r') as f:
        conf = json.load(f)
    
    # Settings for LSTM.
    lm_trainer = Trainer(args, conf, verbose = True) 

    jag = int(args.jagged)
    skipsos = int(args.skipsos)
    outfile = None
    verbose = False
    
    sentence_length = conf["sentence_length"] - 1 - int(args.skipsos) + int(args.bidirectional)

    base_logdir = './runs/'
    id_string = create_identifier_string(lm_trainer.model.name)
    log_dir = os.path.join(base_logdir, id_string)

    writer = SummaryWriter(log_dir)

    train_days = conf['train_files']
    test_days = conf['test_files']

    train_loader, test_loader = data_utils.load_data(train_days, test_days, args, sentence_length)

    for iteration, batch in enumerate(train_loader):
        loss, done = lm_trainer.train_step(batch)
        writer.add_scalar(f'Loss/train_day_{batch["day"][0]}', loss, iteration)
        if done:
            print("Early stopping.")
            break

    for iteration, batch in enumerate(test_loader):
        loss = lm_trainer.eval_step(batch)
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
                                     
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-folder", type=str, help="Path to data files.")
    parser.add_argument('--jagged', action='store_true',
                        help='Whether using sequences of variable length (Input should'
                             'be zero-padded to max_sequence_length.')
    parser.add_argument('--skipsos', action='store_true',
                        help='Whether to skip a start of sentence token.')
    parser.add_argument('--bidir', dest='bidirectional', action='store_true',
                        help='Whether to use bidirectional lstm for lower tier.')
    parser.add_argument('-bs', dest='batch_size', type=int, help='batch size.')
    parser.add_argument("-lstm_layers", nargs='+', type=int, default=[10],
                        help="A list of hidden layer sizes.")
    parser.add_argument('-embed_dim', type=int, default=20,
                        help='Size of embeddings for categorical features.')
    parser.add_argument('--config', type=str, default='config.json', help='JSON configuration file')
    parser.add_argument('--model_dir', type=str, help='Directory to save stats and checkpoints to', default='runs')
    parser.add_argument('--load_from_checkpoint', type=str, help='Checkpoint to resume training from')
    args = parser.parse_args()
    train(args)