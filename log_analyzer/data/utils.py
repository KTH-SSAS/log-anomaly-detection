
"""
Data loading functions
"""

from typing import Iterator
from torch.utils.data import DataLoader, IterableDataset
import torch
import os.path as path
import numpy as np

DEFAULT_HEADERS = [
    "line_number",
    "second",
    "day",
    "user",
    "red",
    "seq_len",
    "start_sentence"
]

class IterableLogDataSet(IterableDataset):

    def __init__(self, filepath, bidirectional, skipsos, jagged, delimiter=' ') -> None:
        super().__init__()
        self.filepath = filepath
        self.delimiter = delimiter

        self.skipsos = skipsos
        self.jag = jagged
        self.bidir = bidirectional
        self.sentence_length = 120 #TODO have this be set in config file

    def parse_lines(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                split_line = line.strip().split(self.delimiter)
                split_line = [int(x) for x in split_line]
                data = torch.LongTensor(split_line)

                endx = data.shape[0] - int(not self.bidir)
                endt = data.shape[0] - int(self.bidir)

                datadict = {
                    'line':     data[0],
                    'second':   data[1],
                    'day':      data[2],
                    'user':     data[3],
                    'red':      data[4],
                    'x':        data[(5+self.jag+self.skipsos):endx],
                    't':        data[(6+self.jag+self.skipsos):endt]
                    }

                if self.jag:
                    datadict['lengths'] = data[5]
                    datadict['mask'] = get_mask(datadict['lengths']-2*self.bidir-self.skipsos, self.sentence_length-2*self.args.bidir)
                    assert np.all(datadict['lengths'] <= datadict['x'].shape[1]), 'Sequence found greater than num_tokens_predicted'
                    assert np.nonzero(datadict['lengths'])[0].shape[0] == datadict['lengths'].shape[0], \
                        'Sequence lengths must be greater than zero.' \
                        'Found zero length sequence in datadict["lengths"]: %s' % datadict['lengths']

                yield datadict

    def __iter__(self):
        return self.parse_lines(self.filepath)


def create_data_loader(filepath, args):
    ds = IterableLogDataSet(filepath, args.bidirectional, args.skipsos, args.jagged)
    return DataLoader(ds, batch_size=args.batch_size)

def load_data(day_train, day_eval, args):

    filepath_train = path.join(args.data_folder, f"{day_train}.txt")
    filepath_eval = path.join(args.data_folder, f"{day_eval}.txt")
    train_loader = create_data_loader(filepath_train, args)
    test_loader = create_data_loader(filepath_eval, args)

    return train_loader, test_loader

def get_mask(lens, num_tokens):
    """
    For masking output of lm_rnn for jagged sequences for correct gradient update.
    Sequence length of 0 will output nan for that row of mask so don't do this.

    :param lens: Numpy vector of sequence lengths
    :param num_tokens: (int) Number of predicted tokens in sentence.
    :return: A numpy array mask MB X num_tokens
             For each row there are: lens[i] values of 1/lens[i]
                                     followed by num_tokens - lens[i] zeros
    """
    mask_template = torch.arange(num_tokens, dtype=torch.float)
    return (mask_template < lens) / lens