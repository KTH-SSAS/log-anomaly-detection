
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

def char_tokens_to_text(tokens):
    characters = [chr(t + 30) for t in tokens]
    string = "".join(characters)
    return characters, string


def translate_line(string, pad_len):
    """

    :param string:
    :param pad_len:
    :return:
    """
    return "0 " + " ".join([str(ord(c) - 30) for c in string]) + " 1 " + " ".join(["0"] * pad_len) + "\n"

class IterableLogDataSet(IterableDataset):

    def __init__(self, filepath, bidirectional, skipsos, jagged, sentence_length, delimiter=' ') -> None:
        super().__init__()
        self.filepath = filepath
        self.delimiter = delimiter

        self.skipsos = skipsos
        self.jag = jagged
        self.bidir = bidirectional
        self.sentence_length = sentence_length

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
                    datadict['length'] = data[5]
                    datadict['mask'] = get_mask(datadict['length']-2*self.bidir-int(self.skipsos), self.sentence_length-2*self.bidir)
                    assert datadict['length'] <= datadict['x'].shape[-1], 'Sequence found greater than num_tokens_predicted'
                    assert datadict['length'] > 0, \
                        'Sequence lengths must be greater than zero.' \
                        f'Found zero length sequence in datadict["lengths"]: {datadict["lengths"]}' 

                yield datadict

    def __iter__(self):
        return self.parse_lines(self.filepath)


def create_data_loader(filepath, args, sentence_length):
    ds = IterableLogDataSet(filepath, args.bidirectional, args.skipsos, args.jagged, sentence_length)
    return DataLoader(ds, batch_size=args.batch_size)

def load_data(train_file, eval_file, args, sentence_length):

    filepath_train = path.join(args.data_folder, train_file)
    filepath_eval = path.join(args.data_folder, eval_file)
    train_loader = create_data_loader(filepath_train, args, sentence_length)
    test_loader = create_data_loader(filepath_eval, args, sentence_length)

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