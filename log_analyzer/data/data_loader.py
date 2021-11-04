
"""
Data loading functions
"""

from log_analyzer.config.trainer_config import DataConfig
from typing import Iterator
from torch.utils.data import DataLoader, IterableDataset
import torch
import os.path as path
import numpy as np
import dask.dataframe as dd
import pandas as pd
from log_analyzer.application import Application

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

        if type(filepaths) is str:
            filepaths = [filepaths]
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
                        if self.bidir:
                            datadict['t'] = datadict['t'].clone().detach()
                            # Mask out the last token of the target sequence
                            datadict['t'][data[5]-1] = 0
                            # include eos token in length
                            datadict['length'] += 1
                        datadict['mask'] = get_mask(
                            datadict['length']-2*self.bidir-int(self.skipsos), self.sentence_length-2*self.bidir)
                        assert datadict['length'] <= datadict['x'].shape[-1], 'Sequence found greater than num_tokens_predicted'
                        assert datadict['length'] > 0, \
                            'Sequence lengths must be greater than zero.' \
                            f'Found zero length sequence in datadict["lengths"]: {datadict["lengths"]}'

                    yield datadict

    def __iter__(self):
        return self.parse_lines()


def load_data_tiered(data_folder, train_files, test_files, batch_size, bidir, skipsos, jagged, sentence_length, num_steps, context_layers):
    def create_data_loader(filepath):
        data_handler = OnlineLMBatcher(filepath,
                                       sentence_length,
                                       context_layers,
                                       skipsos,
                                       jagged,
                                       bidir,
                                       batch_size=batch_size,
                                       num_steps=num_steps,
                                       delimiter=" ")
        return data_handler
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_data_loader(filepaths_train)
    test_loader = create_data_loader(filepaths_eval)
    return train_loader, test_loader

def load_data_tiered_trans(data_folder, train_files, test_files, batch_size, bidir, skipsos, jagged, sentence_length, num_steps, context_model_dim, context_input_dimension):
    def create_data_loader(filepath):
        data_handler = TieredOnlineLMBatcher(filepath,
                                       sentence_length,
                                       context_model_dim,
                                       skipsos,
                                       jagged,
                                       bidir,
                                       context_input_dimension,
                                       batch_size=batch_size,
                                       num_steps=num_steps,
                                       delimiter=" ")
        return data_handler
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_data_loader(filepaths_train)
    test_loader = create_data_loader(filepaths_eval)
    return train_loader, test_loader

def load_data(data_folder, train_files, test_files, batch_size, bidir, skipsos, jagged, sentence_length):
    def create_data_loader(filepath):
        dataset = IterableLogDataSet(
            filepath, bidir, skipsos, jagged, sentence_length)
        data_handler = DataLoader(dataset, batch_size=batch_size)
        return data_handler
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_data_loader(filepaths_train)
    test_loader = create_data_loader(filepaths_eval)

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
    if Application.instance().using_cuda:
        mask_template = torch.arange(num_tokens, dtype=torch.float).cuda()
    else:
        mask_template = torch.arange(num_tokens, dtype=torch.float)
    return (mask_template < lens) / lens


class OnlineLMBatcher:
    """
    For use with tiered_lm.py. Batcher keeps track of user states in upper tier RNN.
    """

    def __init__(self, filepaths, sentence_length, context_size, skipsos, jagged, bidir,
                 batch_size=100, num_steps=5, delimiter=" ",
                 skiprows=0):
        self.sentence_length = sentence_length
        self.context_size = context_size if type(context_size) is list else [context_size]
        self.jagged = jagged
        self.skipsos = skipsos
        self.bidir = bidir
        self.delimiter = delimiter  # delimiter for input file
        self.init_mb_size = batch_size
        self.init_num_steps = num_steps
        self.mb_size = batch_size  # the number of users in a batch
        self.num_steps = num_steps  # The number of log lines for each user in a batch
        self.user_logs = {}
        self.flush = False
        self.skip_file = False # Used by the trainer to signal if the rest of the current file should be skipped when flush is reached
        self.empty = False
        self.staggler_num_steps = 1
        # the list of users whose saved log lines are greater than or equal to the self.num_steps
        self.users_ge_num_steps = []
        self.filepaths = filepaths
        self.saved_lstm = {}
        self.skiprows = skiprows
        self.cuda = Application.instance().using_cuda

    def __iter__(self):
        for datafile in self.filepaths:
            self.flush = False
            self.empty = False
            self.mb_size = self.init_mb_size
            self.num_steps = self.init_num_steps
            self.skip_file = False

            with open(datafile, 'r') as f:
                for i in range(self.skiprows):
                    _ = f.readline()

                while True:
                    output = []
                    if self.skip_file == True:
                        # Skip the rest of the current file, because it is flush and
                        # we're currently training (see train_loop.py)
                        break
                    while output == []:
                        if self.flush == False:
                            l = f.readline()
                            if l == '':
                                self.flush = True
                            else:
                                split_line = l.strip().split(self.delimiter)
                                rowtext = [int(k) for k in split_line]
                                user = int(rowtext[3])

                                if self.user_logs.get(user) is None:
                                    self.user_logs[user] = []
                                    if self.cuda:
                                        self.saved_lstm[user] = (
                                            torch.zeros((self.context_size[0])).cuda(),
                                            torch.zeros((len(self.context_size), self.context_size[0])).cuda(),
                                            torch.zeros((len(self.context_size), self.context_size[0])).cuda()
                                            )
                                    else:
                                        self.saved_lstm[user] = (
                                            torch.zeros((self.context_size[0])),
                                            torch.zeros((len(self.context_size), self.context_size[0])),
                                            torch.zeros((len(self.context_size), self.context_size[0])))
                                self.user_logs[user].append(rowtext)
                                if user not in self.users_ge_num_steps and len(self.user_logs[user]) >= self.num_steps:
                                    self.users_ge_num_steps.append(user)

                        # Before the data loader read the last line of the log.
                        if len(self.users_ge_num_steps) >= self.mb_size and self.flush == False:
                            output, ctxt_vector, h_state, c_state = self.load_lines()

                        # When the data loader read the last line of the log.
                        elif len(self.users_ge_num_steps) > 0 and self.flush == True:
                            output, ctxt_vector, h_state, c_state = self.load_lines()

                        # Activate the staggler mode.
                        elif len(self.users_ge_num_steps) == 0 and self.flush == True:
                            if self.num_steps == self.staggler_num_steps:
                                self.empty = True
                                break
                            self.mb_size = self.num_steps * self.mb_size
                            self.num_steps = self.staggler_num_steps

                    if self.empty == True:
                        break

                    output = torch.Tensor(output).long()
                    batch = torch.transpose(output, 0, 1)
                    endx = batch.shape[2] - int(not self.bidir)
                    endt = batch.shape[2] - int(self.bidir)
                    datadict = {'line': batch[:, :, 0],
                                'second': batch[:, :, 1],
                                'day': batch[:, :, 2],
                                'user': batch[:, :, 3],
                                'red': batch[:, :, 4],
                                'x': batch[:, :, 5 + self.jagged + self.skipsos:endx],
                                't': batch[:, :, 6 + self.jagged + self.skipsos:endt],
                                'context_vector': ctxt_vector,
                                'c_state_init': torch.transpose(h_state, 0, 1),
                                'h_state_init': torch.transpose(c_state, 0, 1)}  # state_triple['h_state_init']}
                    if self.jagged:
                        if self.cuda:
                            datadict['length'] = torch.LongTensor(
                                batch[:, :, 5] - int(self.skipsos)).cuda()
                            datadict['mask'] = torch.empty(
                                datadict['length'].shape[0], datadict['x'].shape[1], datadict['x'].shape[-1] - 2 * self.bidir).cuda()
                        else:
                            datadict['length'] = torch.LongTensor(
                                batch[:, :, 5] - int(self.skipsos))
                            datadict['mask'] = torch.empty(
                                datadict['length'].shape[0], datadict['x'].shape[1], datadict['x'].shape[-1] - 2 * self.bidir)

                        for i, seq_len_matrix in enumerate(datadict['length']):
                            for j, seq_length in enumerate(seq_len_matrix):
                                datadict['mask'][i, j, :] = get_mask(
                                    seq_length.view(-1, 1) - 2 * self.bidir, self.sentence_length - 2 * self.bidir)
                    yield datadict

    def load_lines(self):
        output = []
        if self.cuda:
            ctxt_vector = torch.tensor([]).cuda()
            h_state = torch.tensor([]).cuda()
            c_state = torch.tensor([]).cuda()
        else:
            ctxt_vector = torch.tensor([])
            h_state = torch.tensor([])
            c_state = torch.tensor([])
        for user in self.users_ge_num_steps[:self.mb_size]:
            output.append(self.user_logs[user][0:self.num_steps])
            self.user_logs[user] = self.user_logs[user][self.num_steps:]
            if len(self.user_logs[user]) < self.num_steps:
                self.users_ge_num_steps.remove(user)
            ctxt_vector = torch.cat((ctxt_vector, torch.unsqueeze(
                self.saved_lstm[user][0], dim=0)), dim=0)
            h_state = torch.cat((h_state, torch.unsqueeze(
                self.saved_lstm[user][1], dim=0)), dim=0)
            c_state = torch.cat((c_state, torch.unsqueeze(
                self.saved_lstm[user][2], dim=0)), dim=0)
        return output, ctxt_vector, h_state, c_state

    def update_state(self, ctxt_vectors, h_states, c_states):
        ctxt_vectors = ctxt_vectors.data
        h_states = torch.transpose(h_states.data, 0, 1)
        c_states = torch.transpose(c_states.data, 0, 1)
        for usr, ctxt_v, h_state, c_state in zip(self.users_ge_num_steps[:self.mb_size], ctxt_vectors, h_states, c_states):
            self.saved_lstm[usr] = (ctxt_v, h_state, c_state)


class TieredOnlineLMBatcher(OnlineLMBatcher):
    
    def __init__(self, filepaths, sentence_length, context_model_dim, skipsos, jagged, bidir, context_input_dimension,
                 batch_size=100, num_steps=5, delimiter=" ",
                 skiprows=0):
        # super().__init__(filepaths, sentence_length, context_size, skipsos, jagged, bidir,
        #          batch_size, num_steps, delimiter, skiprows)
        
        self.sentence_length = sentence_length
        self.jagged = jagged
        self.skipsos = skipsos
        self.bidir = bidir
        self.delimiter = delimiter  # delimiter for input file
        self.init_mb_size = batch_size
        self.init_num_steps = num_steps
        self.mb_size = batch_size  # the number of users in a batch
        self.num_steps = num_steps  # The number of log lines for each user in a batch
        self.user_logs = {}
        self.flush = False
        self.skip_file = False # Used by the trainer to signal if the rest of the current file should be skipped when flush is reached
        self.empty = False
        self.staggler_num_steps = 1
        # the list of users whose saved log lines are greater than or equal to the self.num_steps
        self.users_ge_num_steps = []
        self.filepaths = filepaths
        self.skiprows = skiprows
        self.cuda = Application.instance().using_cuda
        self.saved_ctxt = {}
        self.context_model_dim = context_model_dim
        self.context_input_dimension =  context_input_dimension

    def __iter__(self):
        for datafile in self.filepaths:
            self.flush = False
            self.empty = False
            self.mb_size = self.init_mb_size
            self.num_steps = self.init_num_steps
            self.skip_file = False

            with open(datafile, 'r') as f:
                for i in range(self.skiprows):
                    _ = f.readline()

                while True:
                    output = []
                    if self.skip_file == True:
                        # Skip the rest of the current file, because it is flush and
                        # we're currently training (see train_loop.py)
                        break
                    while output == []:
                        if self.flush == False:
                            l = f.readline()
                            if l == '':
                                self.flush = True
                            else:
                                split_line = l.strip().split(self.delimiter)
                                rowtext = [int(k) for k in split_line]
                                user = int(rowtext[3])

                                if self.user_logs.get(user) is None:
                                    self.user_logs[user] = []
                                    if self.cuda:
                                        self.saved_ctxt[user] = [
                                            torch.zeros(self.context_model_dim).cuda(),
                                            torch.tensor([]).cuda(),
                                            0
                                            ]
                                    else:
                                        self.saved_ctxt[user] = [
                                            torch.zeros(self.context_model_dim),
                                            torch.tensor([]),
                                            0
                                            ]
                                self.user_logs[user].append(rowtext)
                                if user not in self.users_ge_num_steps and len(self.user_logs[user]) >= self.num_steps:
                                    self.users_ge_num_steps.append(user)

                        # Before the data loader read the last line of the log.
                        if len(self.users_ge_num_steps) >= self.mb_size and self.flush == False:
                            output, ctxt_vector, history, history_length = self.load_lines()

                        # When the data loader read the last line of the log.
                        elif len(self.users_ge_num_steps) > 0 and self.flush == True:
                            output, ctxt_vector, history, history_length= self.load_lines()

                        # Activate the staggler mode.
                        elif len(self.users_ge_num_steps) == 0 and self.flush == True:
                            if self.num_steps == self.staggler_num_steps:
                                self.empty = True
                                break
                            self.mb_size = self.num_steps * self.mb_size
                            self.num_steps = self.staggler_num_steps

                    if self.empty == True:
                        break

                    output = torch.Tensor(output).long()
                    batch = torch.transpose(output, 0, 1)
                    endx = batch.shape[2] - int(not self.bidir)
                    endt = batch.shape[2] - int(self.bidir)
                    datadict = {'line': batch[:, :, 0],
                                'second': batch[:, :, 1],
                                'day': batch[:, :, 2],
                                'user': batch[:, :, 3],
                                'red': batch[:, :, 4],
                                'x': batch[:, :, 5 + self.jagged + self.skipsos:endx],
                                't': batch[:, :, 6 + self.jagged + self.skipsos:endt],
                                'context_vector': ctxt_vector,
                                'history': history,
                                'history_length': history_length
                                # 'c_state_init': torch.transpose(h_state, 0, 1),
                                # 'h_state_init': torch.transpose(c_state, 0, 1)
                                }  # state_triple['h_state_init']}
                    if self.jagged:
                        if self.cuda:
                            datadict['length'] = torch.LongTensor(
                                batch[:, :, 5] - int(self.skipsos)).cuda()
                            datadict['mask'] = torch.empty(
                                datadict['length'].shape[0], datadict['x'].shape[1], datadict['x'].shape[-1] - 2 * self.bidir).cuda()
                        else:
                            datadict['length'] = torch.LongTensor(
                                batch[:, :, 5] - int(self.skipsos))
                            datadict['mask'] = torch.empty(
                                datadict['length'].shape[0], datadict['x'].shape[1], datadict['x'].shape[-1] - 2 * self.bidir)

                        for i, seq_len_matrix in enumerate(datadict['length']):
                            for j, seq_length in enumerate(seq_len_matrix):
                                datadict['mask'][i, j, :] = get_mask(
                                    seq_length.view(-1, 1) - 2 * self.bidir, self.sentence_length - 2 * self.bidir)
                    yield datadict

    def load_lines(self):
        output = []
        hist_lst = []
        hist_lengths = []
        hist_dimension = 0
        if self.cuda:
            ctxt_vector = torch.tensor([]).cuda()
            history = torch.tensor([]).cuda()
        else:
            ctxt_vector = torch.tensor([])
            history = torch.tensor([])
        for user in self.users_ge_num_steps[:self.mb_size]:
            output.append(self.user_logs[user][0:self.num_steps])
            self.user_logs[user] = self.user_logs[user][self.num_steps:]

            ctxt_vector = torch.cat((ctxt_vector, torch.unsqueeze(
                self.saved_ctxt[user][0], dim=0)), dim=0)
            hist_lst.append(torch.unsqueeze(self.saved_ctxt[user][1], dim=0))
            hist_lengths.append(self.saved_ctxt[user][2])
            hist_dimension = max(self.saved_ctxt[user][1].shape[-1], hist_dimension)
        max_length = max(hist_lengths)
        for idx, hist in enumerate(hist_lst):
            if hist_lengths[idx] == max_length:
                hist_lst[idx] = hist
            elif hist_lengths[idx] == 0:        
                if self.cuda:
                    hist_lst[idx] = torch.zeros(1, max_length, hist_dimension).cuda()
                else:
                    hist_lst[idx] = torch.zeros(1, max_length, hist_dimension)
            else:
                if self.cuda:
                    hist_lst[idx] = torch.cat((hist, torch.zeros(1, max_length - hist_lengths[idx], hist_dimension)), dim = 1).cuda()
                else:
                    hist_lst[idx] = torch.cat((hist, torch.zeros(1, max_length - hist_lengths[idx], hist_dimension)), dim = 1)
        history = torch.cat((hist_lst), dim=0)

        return output, ctxt_vector, history, hist_lengths

    def update_state(self, ctxt_vectors, ctxt_history):
        ctxt_vectors = ctxt_vectors.data
        ctxt_history = ctxt_history.data
        remove_usr = []
        for usr, ctxt_v, history in zip(self.users_ge_num_steps[:self.mb_size], ctxt_vectors, ctxt_history):
            self.saved_ctxt[usr] = [ctxt_v, history, history.shape[0]]
            remove_usr.append(usr)
        for usr in remove_usr:
            if len(self.user_logs[usr]) < self.num_steps:
                self.users_ge_num_steps.remove(usr)
