from argparse import ArgumentParser
import argparse
import numpy as np
import os
import json
import log_analyzer.data.utils as data_utils
import log_analyzer.model.lstm as lstms
from log_analyzer import trainer
from log_analyzer import evaluator

"""
Entrypoint script for training
"""

def train(args):

    # Read a config file.   
    with open(args.config, 'r') as f:
        conf = json.load(f)
    
    # Settings for LSTM.
    model, criterion,  optimizer, scheduler, early_stopping, cuda = trainer.training_settings(args, conf, verbose = True) 

    jag = int(args.jagged)
    skipsos = int(args.skipsos)

    for i in range(len(conf['test_files'][:-1])):
        train_day = conf['test_files'][i] # Should be better probably, to be able to train with multiple days / Thank you for this nice change! I agree :)
        test_day = conf['test_files'][i+1]

        train_loader, test_loader = data_utils.load_data(train_day, test_day, args)

        for batch in train_loader:
            model = trainer.train_model(batch, model, criterion, optimizer, scheduler, early_stopping, cuda, args.jagged)
        for batch_num, batch in enumerate(test_loader):
            evaluator.evaluate_model(batch_num, batch, test_day, model, criterion, cuda, args.jagged, verbose = True) 
                                     
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