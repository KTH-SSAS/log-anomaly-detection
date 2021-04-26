
"""
Data loading functions
"""

from typing import Iterator
from torch.utils.data import DataLoader, IterableDataset
import torch
import os.path as path
import numpy as np
import dask.dataframe as dd
import pandas as pd


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

    def __init__(self, filepaths, bidirectional, skipsos, jagged, sentence_length, delimiter=' ') -> None:
        super().__init__()
        self.delimiter = delimiter

        self.skipsos = skipsos
        self.jag = jagged
        self.bidir = bidirectional
        self.sentence_length = sentence_length

        self.filepaths = filepaths

    def parse_lines(self):
        for datafile in self.filepaths:
            with open(datafile, 'r') as f:
                for line in f:
                    split_line = line.strip().split(self.delimiter)
                    split_line = [int(x) for x in split_line]
                    data = torch.LongTensor(split_line)

                    endx = data.shape[0] - int(not self.bidir)
                    endt = data.shape[0] - int(self.bidir)

                    datadict = {
                        'dayfile':  datafile,
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
                        datadict['mask'] = get_mask(
                            datadict['length']-2*self.bidir-int(self.skipsos), self.sentence_length-2*self.bidir)
                        assert datadict['length'] <= datadict['x'].shape[-1], 'Sequence found greater than num_tokens_predicted'
                        assert datadict['length'] > 0, \
                            'Sequence lengths must be greater than zero.' \
                            f'Found zero length sequence in datadict["lengths"]: {datadict["lengths"]}'

                    yield datadict

    def __iter__(self):
        return self.parse_lines()


def create_data_loader(filepath, args, sentence_length, num_steps):
    if args.tiered:
        data_handler = OnlineLMBatcher(filepath, sentence_length, args.context_layers, args.skipsos, args.jagged, args.bidirectional, 
                                        batch_size=args.batch_size, num_steps=num_steps, delimiter=" ")
    else:
        ds = IterableLogDataSet(filepath, args.bidirectional,
                                args.skipsos, args.jagged, sentence_length)
        data_handler = DataLoader(ds, batch_size=args.batch_size)
    return data_handler


def load_data(train_files, eval_files, args, sentence_length, num_steps = 3):

    filepaths_train = [path.join(args.data_folder, f) for f in train_files]
    filepaths_eval = [path.join(args.data_folder, f) for f in eval_files]
    train_loader = create_data_loader(filepaths_train, args, sentence_length, num_steps)
    test_loader = create_data_loader(filepaths_eval, args, sentence_length, num_steps)

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

class OnlineLMBatcher: 
    def __init__(self, file_path, sentence_length, context_size, skipsos, jagged, bidir, batch_size=100, num_steps=3, delimiter=" "):
        #sentence len on json: 12(word), 122(char). weirdly, actual length is 12(word), 123(char), so I added jagged.
        self.sentence_length = sentence_length
        sentence_length = sentence_length + 1 + int(skipsos) - int(bidir) + int(jagged)
        cols = ['line', 'second', 'day', 'user', 'red'] + [f'x_{i}' for i in range(sentence_length)]   
        self.day_df = dd.read_csv(file_path, names=cols, sep = ' ', blocksize=25e3)
        self.user_id = [] # set()
        self.lst_avail_id = []
        self.pre_lst_avail_id = []
        self.df_id = {}
        self.len_id = {}
        self.saved_lstm = {}
        self.context_size = context_size
        self.sel_part = 0
        self.current_num_batch = 0
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.staggler_num_steps = 1
        self.jagged = jagged
        self.skipsos = skipsos
        self.bidir = bidir
        self.empty = False
        
    def filter_partition(self):
        partition = self.day_df.get_partition(self.sel_part).compute()
        current_ids = partition.user.drop_duplicates().tolist()
        for c_id in current_ids:
            if c_id not in self.user_id:
                self.df_id[c_id] = None
                self.saved_lstm[c_id] = (torch.zeros((self.context_size[0])),\
                                         torch.zeros((len(self.context_size), self.context_size[0])),\
                                         torch.zeros((len(self.context_size), self.context_size[0])))
            self.df_id[c_id] = pd.concat([self.df_id[c_id], partition[partition.user == c_id]], axis=0)
            self.len_id[c_id] = len(self.df_id[c_id])

        self.user_id = current_ids + [usr for usr in self.user_id if usr not in current_ids]
        self.sel_part += 1   

    def update_len(self):
        self.lst_avail_id = []
        self.current_num_batch = 0
        for j in self.user_id:
            above_num_steps = self.len_id[j] >= self.num_steps
            self.current_num_batch += above_num_steps
            if above_num_steps and j not in self.lst_avail_id:
                self.lst_avail_id.append(j)
    
    def __iter__(self):
        while not self.empty:
            output = []
            datadict = {}
            ctxt_vector = torch.tensor([])
            h_state = torch.tensor([])
            c_state = torch.tensor([])
            while output == []:
                if self.current_num_batch < self.batch_size and self.sel_part < self.day_df.npartitions: # Read a new partition
                    self.filter_partition()
                    self.update_len()
                elif self.current_num_batch == 0 and self.sel_part == self.day_df.npartitions: # Activate staggler mode
                    self.batch_size = self.batch_size * self.num_steps
                    self.num_steps = self.staggler_num_steps
                    self.sel_part += 1
                    self.update_len()
                elif self.current_num_batch > 0: # Output data
                    for j in self.lst_avail_id[:self.batch_size]:
                        output.append(self.df_id[j].iloc[0:self.num_steps].values)                    
                        ctxt_vector = torch.cat((ctxt_vector, torch.unsqueeze(self.saved_lstm[j][0], dim = 0)), dim = 0)                    
                        h_state = torch.cat((h_state, torch.unsqueeze(self.saved_lstm[j][1], dim = 0)), dim = 0)
                        c_state = torch.cat((c_state, torch.unsqueeze(self.saved_lstm[j][2], dim = 0)), dim = 0)
                        self.df_id[j] = self.df_id[j].iloc[self.num_steps:, :]
                        self.len_id[j] = len(self.df_id[j])
                    self.pre_lst_avail_id = self.lst_avail_id
                    self.update_len()
                    output = torch.tensor(output).long()
                    batch = torch.transpose(output, 0, 1)
                    endx = batch.shape[2] - int(not self.bidir)
                    endt = batch.shape[2] - int(self.bidir)
                    datadict = {'line': batch[:, :, 0],
                                'second': batch[:, :, 1],
                                'day': batch[:, :, 2],
                                'user': batch[:, :, 3],
                                'red': batch[:, :, 4],
                                'x': [batch[0, :, 5 + self.jagged + self.skipsos:endx]] * self.num_steps,
                                't': [batch[0, :, 6 + self.jagged + self.skipsos:endt]] * self.num_steps,
                                'context_vector': ctxt_vector, #,['context_vector'],
                                'c_state_init': torch.transpose(h_state, 0,1), #state_triple['c_state_init'],
                                'h_state_init': torch.transpose(c_state, 0,1)} #state_triple['h_state_init']}
                    if self.jagged:
                        datadict['length'] = [batch[0, :, 5] - int(self.skipsos)] * self.num_steps
                        datadict['mask'] = [get_mask(seq_length.view(-1,1) - 2 * self.bidir, self.sentence_length - 2 * self.bidir) for
                                             seq_length in datadict['length']]
                else: # Empty dataset.
                    self.empty = True
                    return None

            yield datadict
    
    def update_state(self, ctxt_vectors, h_states, c_states):
        ctxt_vectors = ctxt_vectors.data
        h_states = torch.transpose(h_states.data, 0,1)
        c_states = torch.transpose(c_states.data, 0,1)
        for usr, ctxt_v, h_state, c_state in zip(self.pre_lst_avail_id[:self.batch_size], ctxt_vectors, h_states, c_states):
            self.saved_lstm[usr] = (ctxt_v, h_state, c_state) 