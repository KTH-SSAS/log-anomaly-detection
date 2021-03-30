from argparse import ArgumentParser
import argparse
import numpy as np
import log_analyzer.data.utils as data_utils

"""
Entrypoint script for training
"""

def train(args):
    train_loader, test_loader = data_utils.load_data(0, 1, args)

    jag = int(args.jagged)
    skipsos = int(args.skipsos)

    for batch in train_loader:

        pass
        
    pass

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
    args = parser.parse_args()
    train(args)