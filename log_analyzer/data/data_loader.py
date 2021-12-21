"""Data loading functions."""

import os.path as path
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from log_analyzer.application import Application

DEFAULT_HEADERS = [
    "line_number",
    "second",
    "day",
    "user",
    "red",
    "seq_len",
    "start_sentence",
]


def get_mask(lens, max_len=None):
    """For masking output of language model for jagged sequences for correct gradient
    update. Sequence length of 0 will output nan for that row of mask so don't
    do this.

    :param lens: (int) sequence length for this sequence
    :param max_len: (int) Number of predicted tokens in longest sequence in batch. Defaults to lens if not provided
    :return: A numpy array mask of length max_len. There are lens[i] values of 1/lens followed by max_len - lens zeros
    """

    num_tokens = lens if max_len is None else max_len

    mask_template = torch.arange(num_tokens, dtype=torch.float)
    if Application.instance().using_cuda:
        mask_template = mask_template.cuda()

    return (mask_template < lens) / lens


def parse_line(line, jag, bidir, skipsos, delimiter=" "):
    skipeos = True
    split_line = line.strip().split(delimiter)
    split_line = [int(x) for x in split_line]
    data = torch.LongTensor(split_line)

    metadata_offset = 5

    if jag:
        length = data[metadata_offset].item()
    else:
        length = data.shape[0] - metadata_offset - int(skipeos) - int(skipsos)

    offset = int(jag) + int(skipsos)

    input_start = metadata_offset + offset
    input_end = input_start + length

    target_start = input_start + 1
    target_end = input_end + 1

    if bidir:
        input_end += 1
        target_end -= 1

    datadict = {
        "line": data[0],
        "second": data[1],
        "day": data[2],
        "user": data[3],
        "red": data[4],
        "input": data[input_start:input_end],
        "target": data[target_start:target_end],
    }

    if jag:  # Input is variable length
        length = datadict["input"].shape[0]
        datadict["length"] = torch.LongTensor([length])
        datadict["mask"] = get_mask(length - 2 * bidir)
        assert length <= datadict["input"].shape[-1], "Sequence found greater than num_tokens_predicted"
        assert length > 0, (
            "Sequence lengths must be greater than zero."
            f'Found zero length sequence in datadict["lengths"]: {datadict["lengths"]}'
        )

    return datadict


def parse_multiple_files(filepaths, jag, bidir, skipsos, raw_lines=False):
    for datafile in filepaths:
        with open(datafile, "r") as f:
            for line in f:
                if raw_lines:
                    yield line
                else:
                    yield parse_line(line, jag, bidir, skipsos)


class LogDataset:
    def __init__(self, filepaths, bidirectional, skipsos, jagged, delimiter=" ") -> None:
        super().__init__()
        self.delimiter = delimiter

        self.skipsos = skipsos
        self.jag = jagged
        self.bidir = bidirectional

        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths


class MapLogDataset(LogDataset, Dataset):
    """Provides data via __getitem__, allowing arbitrary data entries to be
    accessed via index."""

    def __init__(self, filepaths, bidirectional, skipsos, jagged, delimiter=" ") -> None:
        super().__init__(filepaths, bidirectional, skipsos, jagged, delimiter)

        self.loglines = []
        iterator = parse_multiple_files(self.filepaths, jagged, bidirectional, skipsos, raw_lines=True)

        self.loglines.extend(iterator)

    def __getitem__(self, index):
        log_line = self.loglines[index]
        parsed_line = parse_line(log_line, self.jag, self.bidir, self.skipsos)
        return parsed_line

    def __len__(self):
        return len(self.loglines)


class IterableLogDataset(LogDataset, IterableDataset):
    """Provides data via __iter__, allowing data to be accessed in order
    only."""

    def __init__(self, filepaths, bidirectional, skipsos, jagged, delimiter=" ") -> None:
        super().__init__(filepaths, bidirectional, skipsos, jagged, delimiter)

    def __iter__(self):
        return parse_multiple_files(self.filepaths, self.jag, self.bidir, self.skipsos)


class LogDataLoader(DataLoader):
    """Wrapper class around torch's DataLoader, used for non-tiered data
    loading.

    Provides a function to split the batch provided by the data loader.
    """

    def __init__(self, dataset, batch_size, shuffle, collate_fn):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        self.cuda = Application.instance().using_cuda

    def split_batch(self, batch: dict):
        """Splits a batch into variables containing relevant data."""
        X = batch["input"]
        Y = batch["target"]

        # Optional fields
        L = batch.get("length")
        M = batch.get("mask")

        if self.cuda:
            X = X.cuda()
            Y = Y.cuda()
            if M is not None:
                M = M.cuda()

        split_batch = {
            "X": X,
            "Y": Y,
            "L": L,
            "M": M,
        }

        # Grab evaluation data
        split_batch["user"] = batch["user"]
        split_batch["second"] = batch["second"]
        split_batch["red_flag"] = batch["red"]

        # Grab any extra parts (model_info) as necessary - for future integration with the tiered models
        if "h_state_init" in batch:
            # Tiered LSTM
            C_V = batch["context_vector"]
            C_H = batch["c_state_init"]
            C_C = batch["h_state_init"]

            if self.cuda:
                C_V = C_V.cuda()
                C_H = C_H.cuda()
                C_C = C_C.cuda()

            split_batch["model_info"] = (C_V, C_H, C_C)
        elif "history_length" in batch:
            # Tiered Transformer
            C_V = batch["context_vector"]
            C_H = batch["history"]
            H_L = batch["history_length"]

            if self.cuda:
                C_V = C_V.cuda()
                C_H = C_H.cuda()

            split_batch["model_info"] = (C_V, C_H, H_L)
        return split_batch


def load_data_tiered(
    data_folder,
    train_files,
    test_files,
    batch_size,
    bidir,
    skipsos,
    jagged,
    sentence_length,
    num_steps,
    context_layers,
):
    def create_tiered_data_loader(filepath):
        data_handler = TieredLSTMBatcher(
            filepath,
            sentence_length,
            context_layers,
            skipsos,
            jagged,
            bidir,
            batch_size=batch_size,
            num_steps=num_steps,
            delimiter=" ",
        )
        return data_handler

    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_tiered_data_loader(filepaths_train)
    test_loader = create_tiered_data_loader(filepaths_eval)
    return train_loader, test_loader


def load_data_tiered_trans(
    data_folder,
    train_files,
    test_files,
    batch_size,
    bidir,
    skipsos,
    jagged,
    sentence_length,
    num_steps,
    context_model_dim,
    context_input_dimension,
    shift_window,
):
    def create_tiered_data_loader(filepath):
        data_handler = TieredTransformerBatcher(
            filepath,
            sentence_length,
            context_model_dim,
            skipsos,
            jagged,
            bidir,
            context_input_dimension,
            shift_window=shift_window,
            batch_size=batch_size,
            num_steps=num_steps,
            delimiter=" ",
        )
        return data_handler

    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_tiered_data_loader(filepaths_train)
    test_loader = create_tiered_data_loader(filepaths_eval)
    return train_loader, test_loader


def create_data_loaders(filepath, batch_size, bidir, skipsos, jagged, max_len, shuffle=False, dataset_split=None):
    """Creates and returns 2 data loaders.

    If dataset_split is not provided the second data loader is instead
    set to None.
    """

    def collate_fn(data, jagged=False):
        """Pads the input fields to the length of the longest sequence in the
        batch."""
        batch = {}

        for key in data[0]:
            batch[key] = []

        for sample in data:
            for key in sample:
                batch[key].append(sample[key])

        if jagged:
            fields_to_pad = ["input", "target", "mask"]
            for key in fields_to_pad:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)

        for key in batch:
            if isinstance(batch[key], list):
                batch[key] = torch.stack(batch[key])

        return batch

    if shuffle or dataset_split is not None:
        dataset = MapLogDataset(filepath, bidir, skipsos, jagged, max_len)
    else:
        dataset = IterableLogDataset(filepath, bidir, skipsos, jagged, max_len)

    # Split the dataset according to the split list
    if dataset_split is not None:
        # Ensure the list has 2 values and sums to 1
        if sum(dataset_split) != 1:
            raise ValueError("Sum of list of splits is not 1.")
        if len(dataset_split) != 2:
            raise ValueError("Split list does not contain exactly 2 values.")
        # Convert splits into lengths as proportion of dataset length
        dataset_split = [int(split_val * len(dataset)) for split_val in dataset_split]
        # Ensure sum of dataset_split is the same as dataset length
        size_diff = len(dataset) - sum(dataset_split)
        dataset_split[0] += size_diff

        datasets = torch.utils.data.random_split(dataset, dataset_split)
    else:
        # Return just a single dataset
        datasets = [dataset, None]

    collate = partial(collate_fn, jagged=jagged)
    data_handlers = []
    for dataset in datasets:
        if dataset is None:
            data_handlers.append(None)
        else:
            data_handlers.append(LogDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate))
    return data_handlers


def load_data(
    data_folder,
    train_files,
    test_files,
    batch_size,
    bidir,
    skipsos,
    jagged,
    sentence_length,
    train_val_split=[1, 0],
    shuffle_train_data=True,
):
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader, val_loader = create_data_loaders(
        filepaths_train, batch_size, bidir, skipsos, jagged, sentence_length, shuffle_train_data, train_val_split
    )
    test_loader, _ = create_data_loaders(
        filepaths_eval, batch_size, bidir, skipsos, jagged, sentence_length, shuffle=False, dataset_split=None
    )

    return train_loader, val_loader, test_loader


class OnlineLMBatcher:
    """For use with tiered language models.

    Batcher keeps track of user states in upper tier RNN.
    """

    def __init__(
        self,
        filepaths,
        sentence_length,
        skipsos,
        jagged,
        bidir,
        batch_size=100,
        num_steps=5,
        delimiter=" ",
        skiprows=0,
    ):
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
        # Used by the trainer to signal if the rest of the current file should
        # be skipped when flush is reached
        self.skip_file = False
        self.empty = False
        self.staggler_num_steps = 1
        # the list of users whose saved log lines are greater than or equal to
        # the self.num_steps
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

            with open(datafile, "r") as f:
                for i in range(self.skiprows):
                    _ = f.readline()

                while True:
                    output = []
                    if self.skip_file:
                        # Skip the rest of the current file, because it is flush and
                        # we're currently training (see train_loop.py)
                        break
                    while output == []:
                        if not self.flush:
                            # Read in the next line
                            l = f.readline()
                            if l == "":
                                self.flush = True
                            else:
                                split_line = l.strip().split(self.delimiter)
                                rowtext = [int(k) for k in split_line]
                                user = int(rowtext[3])

                                if self.user_logs.get(user) is None:
                                    self.user_logs[user] = []
                                    self.init_saved_model(user)
                                self.user_logs[user].append(rowtext)
                                if user not in self.users_ge_num_steps and len(self.user_logs[user]) >= self.num_steps:
                                    self.users_ge_num_steps.append(user)

                        # Before the data loader read the last line of the log.
                        if len(self.users_ge_num_steps) >= self.mb_size and self.flush == False:
                            output, model_info = self.load_lines()

                        # When the data loader read the last line of the log.
                        elif len(self.users_ge_num_steps) > 0 and self.flush:
                            output, model_info = self.load_lines()

                        # Activate the staggler mode.
                        elif len(self.users_ge_num_steps) == 0 and self.flush:
                            if self.num_steps == self.staggler_num_steps:
                                self.empty = True
                                break
                            self.mb_size = self.num_steps * self.mb_size
                            self.num_steps = self.staggler_num_steps

                    if self.empty:
                        break

                    output = torch.Tensor(output).long()
                    batch = torch.transpose(output, 0, 1)
                    endx = batch.shape[2] - int(not self.bidir)
                    endt = batch.shape[2] - int(self.bidir)
                    datadict = self.gen_datadict(batch, endx, endt, model_info)

                    if self.jagged:
                        datadict["length"] = torch.LongTensor(batch[:, :, 5] - int(self.skipsos))
                        datadict["mask"] = torch.empty(
                            datadict["length"].shape[0],
                            datadict["input"].shape[1],
                            datadict["input"].shape[-1] - 2 * self.bidir,
                        )
                        if self.cuda:
                            datadict["length"] = datadict["length"].cuda()
                            datadict["mask"] = datadict["mask"].cuda()

                        for i, seq_len_matrix in enumerate(datadict["length"]):
                            for j, seq_length in enumerate(seq_len_matrix):
                                datadict["mask"][i, j, :] = get_mask(
                                    seq_length.view(-1, 1).item() - 2 * self.bidir,
                                    self.sentence_length - 2 * self.bidir,
                                )
                    yield datadict

    def init_saved_model(self, user):
        pass

    def load_lines(self):
        pass

    def gen_datadict(self, batch, endx, endt, model_info):
        pass


class TieredLSTMBatcher(OnlineLMBatcher):
    def __init__(
        self,
        filepaths,
        sentence_length,
        context_size,
        skipsos,
        jagged,
        bidir,
        batch_size=100,
        num_steps=5,
        delimiter=" ",
        skiprows=0,
    ):
        super().__init__(filepaths, sentence_length, skipsos, jagged, bidir, batch_size, num_steps, delimiter, skiprows)
        self.context_size = context_size if type(context_size) is list else [context_size]

    def split_batch(self, batch: dict):
        """Splits a batch into variables containing relevant data."""

        X = batch["input"]
        Y = batch["target"]

        # Optional fields
        L = batch.get("length")
        M = batch.get("mask")

        if self.cuda:
            X = X.cuda()
            Y = Y.cuda()
            if M is not None:
                M = M.cuda()

        C_V = batch["context_vector"]
        C_H = batch["c_state_init"]
        C_C = batch["h_state_init"]

        if self.cuda:
            C_V = C_V.cuda()
            C_H = C_H.cuda()
            C_C = C_C.cuda()

        split_batch = {
            "X": X,
            "Y": Y,
            "L": L,
            "M": M,
            "model_info": (C_V, C_H, C_C),
        }

        # Grab evaluation data
        split_batch["user"] = batch["user"]
        split_batch["second"] = batch["second"]
        split_batch["red_flag"] = batch["red"]

        return split_batch

    def init_saved_model(self, user):
        if self.cuda:
            self.saved_lstm[user] = (
                torch.zeros((self.context_size[0])).cuda(),
                torch.zeros((len(self.context_size), self.context_size[0],)).cuda(),
                torch.zeros((len(self.context_size), self.context_size[0],)).cuda(),
            )

        else:
            self.saved_lstm[user] = (
                torch.zeros((self.context_size[0])),
                torch.zeros((len(self.context_size), self.context_size[0],)),
                torch.zeros((len(self.context_size), self.context_size[0],)),
            )

    def gen_datadict(self, batch, endx, endt, model_info):
        ctxt_vector = model_info[0]
        h_state = model_info[1]
        c_state = model_info[2]
        datadict = {
            "line": batch[:, :, 0],
            "second": batch[:, :, 1],
            "day": batch[:, :, 2],
            "user": batch[:, :, 3],
            "red": batch[:, :, 4],
            "input": batch[:, :, 5 + self.jagged + self.skipsos : endx],
            "target": batch[:, :, 6 + self.jagged + self.skipsos : endt],
            "context_vector": ctxt_vector,
            "c_state_init": torch.transpose(h_state, 0, 1),
            "h_state_init": torch.transpose(c_state, 0, 1),
        }
        return datadict

    def load_lines(self):
        output = []
        ctxt_vector = torch.tensor([])
        h_state = torch.tensor([])
        c_state = torch.tensor([])
        if self.cuda:
            ctxt_vector = ctxt_vector.cuda()
            h_state = h_state.cuda()
            c_state = c_state.cuda()
        # Loop over users that have enough lines loaded to be used in a batch
        for user in self.users_ge_num_steps[: self.mb_size]:
            # Add user's lines to the batch
            output.append(self.user_logs[user][0 : self.num_steps])
            # Update user's saved lines
            self.user_logs[user] = self.user_logs[user][self.num_steps :]
            # Remove user from list if it now doesn't have enough lines left to be used in another batch
            if len(self.user_logs[user]) < self.num_steps:
                self.users_ge_num_steps.remove(user)
            # Grab the context information
            ctxt_vector = torch.cat((ctxt_vector, torch.unsqueeze(self.saved_lstm[user][0], dim=0)), dim=0)
            h_state = torch.cat((h_state, torch.unsqueeze(self.saved_lstm[user][1], dim=0)), dim=0)
            c_state = torch.cat((c_state, torch.unsqueeze(self.saved_lstm[user][2], dim=0)), dim=0)
        return output, (ctxt_vector, h_state, c_state)

    def update_state(self, ctxt_vectors, h_states, c_states):
        ctxt_vectors = ctxt_vectors.data
        h_states = torch.transpose(h_states.data, 0, 1)
        c_states = torch.transpose(c_states.data, 0, 1)
        for usr, ctxt_v, h_state, c_state in zip(
            self.users_ge_num_steps[: self.mb_size], ctxt_vectors, h_states, c_states
        ):
            self.saved_lstm[usr] = (ctxt_v, h_state, c_state)


class TieredTransformerBatcher(OnlineLMBatcher):
    def __init__(
        self,
        filepaths,
        sentence_length,
        context_model_dim,
        skipsos,
        jagged,
        bidir,
        context_input_dimension,
        shift_window=500,
        batch_size=100,
        num_steps=5,
        delimiter=" ",
        skiprows=0,
    ):
        super().__init__(
            filepaths, sentence_length, skipsos, jagged, bidir, batch_size, num_steps, delimiter, skiprows,
        )
        # the list of users whose saved log lines are greater than or equal to the self.num_steps
        self.saved_ctxt = {}
        self.context_model_dim = context_model_dim
        self.context_input_dimension = context_input_dimension
        self.shift_window = shift_window

    def split_batch(self, batch: dict):
        """Splits a batch into variables containing relevant data."""

        X = batch["input"]
        Y = batch["target"]

        # Optional fields
        L = batch.get("length")
        M = batch.get("mask")

        if self.cuda:
            X = X.cuda()
            Y = Y.cuda()
            if M is not None:
                M = M.cuda()

        C_V = batch["context_vector"]
        C_H = batch["history"]
        H_L = batch["history_length"]

        if self.cuda:
            C_V = C_V.cuda()
            C_H = C_H.cuda()

        split_batch = {
            "X": X,
            "Y": Y,
            "L": L,
            "M": M,
            "model_info": (C_V, C_H, H_L),
        }

        # Grab evaluation data
        split_batch["user"] = batch["user"]
        split_batch["second"] = batch["second"]
        split_batch["red_flag"] = batch["red"]

        return split_batch

    def init_saved_model(self, user):
        self.saved_ctxt[user] = [torch.zeros(self.context_model_dim), torch.tensor([]), torch.tensor(0)]
        if self.cuda:
            self.saved_ctxt[user] = self.saved_ctxt[user].cuda()

    def gen_datadict(self, batch, endx, endt, model_info):
        ctxt_vector = model_info[0]
        history = model_info[1]
        history_length = model_info[2]
        datadict = {
            "line": batch[:, :, 0],
            "second": batch[:, :, 1],
            "day": batch[:, :, 2],
            "user": batch[:, :, 3],
            "red": batch[:, :, 4],
            "input": batch[:, :, 5 + self.jagged + self.skipsos : endx],
            "target": batch[:, :, 6 + self.jagged + self.skipsos : endt],
            "context_vector": ctxt_vector,
            "history": history,
            "history_length": history_length,
        }
        return datadict

    def load_lines(self):
        output = []
        hist_lst = torch.tensor([])
        hist_lengths = torch.tensor([])
        hist_dimension = 0
        if self.cuda:
            ctxt_vector = torch.tensor([]).cuda()
            history = torch.tensor([]).cuda()
        else:
            ctxt_vector = torch.tensor([])
            history = torch.tensor([])
        self.current_batch_usr = self.users_ge_num_steps[: self.mb_size]
        # Loop over users that have enough lines loaded to be used in a batch
        for user in self.current_batch_usr:
            # Add user's lines to the batch
            output.append(self.user_logs[user][0 : self.num_steps])
            # Update user's saved lines
            self.user_logs[user] = self.user_logs[user][self.num_steps :]
            # Remove user from list if it now doesn't have enough lines left to be used in another batch
            if len(self.user_logs[user]) < self.num_steps:
                self.users_ge_num_steps.remove(user)
            # Grab the context information
            ctxt_vector = torch.cat((ctxt_vector, torch.unsqueeze(self.saved_ctxt[user][0], dim=0)), dim=0)
            hist_lst = torch.cat((hist_lst, torch.unsqueeze(self.saved_ctxt[user][1], dim=0)), dim=0)
            hist_lengths = torch.cat((hist_lengths, torch.unsqueeze(self.saved_ctxt[user][2], dim=0)), dim=0)
            hist_dimension = max(self.saved_ctxt[user][1].shape[-1], hist_dimension)

        device = history.device
        max_length = max(hist_lengths)
        for idx, hist in enumerate(hist_lst):
            if hist_lengths[idx] == max_length:
                hist_lst[idx] = hist
            elif hist_lengths[idx] == 0:
                hist_lst[idx] = torch.zeros(1, max_length, hist_dimension).to(device)
            else:
                hist_lst[idx] = torch.cat(
                    (hist, torch.zeros(1, max_length - hist_lengths[idx], hist_dimension)), dim=1
                ).to(device)

        return output, (ctxt_vector, hist_lst, hist_lengths)

    def update_state(self, ctxt_vectors, ctxt_history):
        ctxt_vectors = ctxt_vectors.data
        ctxt_history = ctxt_history.data
        remove_usr = []
        for usr, ctxt_v, history in zip(self.current_batch_usr, ctxt_vectors, ctxt_history):
            self.saved_ctxt[usr] = [ctxt_v, history[: self.shift_window], history.shape[0]]
            remove_usr.append(usr)


# def char_tokens_to_text(tokens):
#     characters = [chr(t + 30) for t in tokens]
#     string = "".join(characters)
#     return characters, string

# def translate_line(string, pad_len):
#     """
#     :param string:
#     :param pad_len:
#     :return:
#     """
#     return "0 " + " ".join([str(ord(c) - 30) for c in string]) + " 1 " + " ".join(["0"] * pad_len) + "\n"
