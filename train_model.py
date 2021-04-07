from argparse import ArgumentParser
import argparse
import numpy as np
import os
import json
import log_analyzer.data.utils as data_utils
import log_analyzer.model.lstm as lstms
from log_analyzer import trainer

"""
Entrypoint script for training
"""

def train(args):

    # Read a config file.
    if args.jagged:
        json_file = 'lanl_char_config.json'
    else:
        json_file = 'lanl_word_config.json'
    specpath = os.path.join(os.getcwd(), 'notebooks/safekit/features/specs/lm', json_file) # For a short test
    conf = json.load(open(specpath, 'r'))
    
    # Settings for LSTM.
    model, criterion,  optimizer, scheduler, early_stopping, cuda = trainer.training_settings(args, conf, verbose = True) 

    jag = int(args.jagged)
    skipsos = int(args.skipsos)    
    train_loader, test_loader = data_utils.load_data(str(0) + 'head', str(1) + 'head', args) # For a short test, I changed it. This part should be decided depending on how we will generate json files.

    for batch in train_loader:
        model = trainer.train_model(batch, model, criterion, optimizer, scheduler, early_stopping, cuda, args.jagged)
        
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
    args = parser.parse_args()
    train(args)