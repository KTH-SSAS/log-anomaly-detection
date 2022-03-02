"""Data loading functions."""

from functools import partial
from os import path
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from log_analyzer.application import Application
from log_analyzer.tokenizer.tokenizer_neo import Tokenizer

AUTOREGRESSIVE_LM = "lm"
BIDIR_LSTM_LM = "bidir-lm"
MASKED_LM = "masked-lm"

SECONDS_PER_DAY = 86400

DEFAULT_HEADERS = [
    "line_number",
    "second",
    "day",
    "user",
    "red",
    "seq_len",
    "start_sentence",
]


def tokens_to_add(task):
    if task == BIDIR_LSTM_LM:
        add_sos = True
        add_eos = True
    elif task == AUTOREGRESSIVE_LM:
        add_sos = False
        add_eos = False
    else:
        add_sos = False
        add_eos = False

    return add_sos, add_eos


def prepare_datadict(line: str, task: str, tokenizer: Tokenizer) -> dict:

    fields = line.strip().split(",")
    second = int(fields[0])
    user = tokenizer.user_idx(fields[1])

    add_sos, add_eos = tokens_to_add(task)

    # Remove timestamp and red team flag from input
    to_tokenize = line[len(fields[0]) + 1 : -3]
    tokenized_line = tokenizer.tokenize(to_tokenize, add_sos, add_eos)

    day = int(second) // SECONDS_PER_DAY

    datadict = {
        "second": torch.LongTensor([second]),
        "day": torch.LongTensor([day]),
        "red": torch.BoolTensor([int(fields[-1])]),
        "user": torch.LongTensor([user]),
    }

    if task == AUTOREGRESSIVE_LM:
        # Shift the input by one
        data_in = tokenized_line[:-1]
        label = tokenized_line[1:]
    elif task == BIDIR_LSTM_LM:
        # Remove first and last token of label since they are not
        # predicted from two directions
        data_in = tokenized_line
        label = tokenized_line[1:-1]
    elif task == MASKED_LM:
        # Add mask tokens to input
        data_in, label, _ = tokenizer.mask_tokens(tokenized_line)
    else:
        raise RuntimeError("Invalid Task")

    datadict["input"] = torch.LongTensor(data_in)
    datadict["target"] = torch.LongTensor(label)
    datadict["length"] = torch.LongTensor([data_in.shape[0]])

    return datadict


def parse_multiple_files(filepaths: List[str]):
    for datafile in filepaths:
        # Only permit ASCII characters.
        with open(datafile, "r", encoding="ascii") as f:
            for line in f:
                yield line


class LogDataset:
    """Base log dataset class."""

    def __init__(self, filepaths, tokenizer: Tokenizer, task) -> None:
        super().__init__()
        self.tokenizer: Tokenizer = tokenizer
        self.task = task

        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths


class MapLogDataset(LogDataset, Dataset):
    """Provides data via __getitem__, allowing arbitrary data entries to be
    accessed via index."""

    def __init__(self, filepaths, tokenizer, task) -> None:
        super().__init__(filepaths, tokenizer, task)

        self.loglines = []
        iterator = parse_multiple_files(self.filepaths)
        self.loglines.extend(iterator)

    def __getitem__(self, index):
        log_line = self.loglines[index]
        parsed_line = prepare_datadict(log_line, self.task, self.tokenizer)
        return parsed_line

    def __len__(self):
        return len(self.loglines)


class IterableLogDataset(LogDataset, IterableDataset):  # pylint: disable=abstract-method
    """Provides data via __iter__, allowing data to be accessed in order
    only."""

    def __iter__(self):
        for line in parse_multiple_files(self.filepaths):
            yield prepare_datadict(line, self.task, self.tokenizer)


class LogDataLoader(DataLoader):
    """Wrapper class around torch's DataLoader, used for non-tiered data
    loading.

    Provides a function to split the batch provided by the data loader.
    """

    def __init__(self, dataset, batch_size, shuffle, collate_fn):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        self.using_cuda = Application.instance().using_cuda

    def split_batch(self, batch: dict):
        """Splits a batch into variables containing relevant data."""
        X = batch["input"]
        Y = batch["target"]

        # Optional fields
        L = batch.get("length")
        M = batch.get("mask")

        if self.using_cuda:
            X = X.cuda()
            Y = Y.cuda()
            if M is not None:
                M = M.cuda()

        split_batch = {
            "X": X,
            "Y": Y,
            "L": L,
            "M": M,
            "user": batch["user"],
            "second": batch["second"],
            "red_flag": batch["red"],
        }

        # Grab evaluation data

        return split_batch


def load_data_tiered(
    data_folder,
    train_files,
    test_files,
    batch_size,
    tokenizer: Tokenizer,
    task,
    num_steps,
):
    def create_tiered_data_loader(filepath):
        data_handler = TieredLogDataLoader(
            filepath,
            tokenizer,
            task,
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


def create_data_loaders(filepath, batch_size, tokenizer, task, shuffle=False, dataset_split=None):
    """Creates and returns 2 data loaders.

    If dataset_split is not provided the second data loader is instead
    set to None.
    """

    def collate_fn(data, jagged=False, pad_idx=0):
        """Pads the input fields to the length of the longest sequence in the
        batch."""
        batch = {}

        for key in data[0]:
            batch[key] = []

        for sample in data:
            for key in sample:
                batch[key].append(sample[key])

        if jagged:
            fields_to_pad = ["input", "target"]
            for key in fields_to_pad:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=pad_idx)

            batch["mask"] = batch["input"] != pad_idx

        for key, value in batch.items():
            if isinstance(value, list):
                batch[key] = torch.stack(value)

        return batch

    if shuffle or dataset_split is not None:
        dataset = MapLogDataset(filepath, tokenizer, task)
    else:
        dataset = IterableLogDataset(filepath, tokenizer, task)

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
        datasets = (dataset, None)

    collate = partial(collate_fn, jagged=tokenizer.jagged)
    data_handlers = [
        LogDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
        if dataset is not None
        else None
        for dataset in datasets
    ]

    return data_handlers


def load_data(
    data_folder,
    train_files,
    test_files,
    batch_size,
    tokenizer: Tokenizer,
    task,
    train_val_split=(1, 0),
    shuffle_train_data=True,
):
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader, val_loader = create_data_loaders(
        filepaths_train, batch_size, tokenizer, task, shuffle_train_data, train_val_split
    )
    test_loader, _ = create_data_loaders(filepaths_eval, batch_size, tokenizer, task, shuffle=False, dataset_split=None)

    return train_loader, val_loader, test_loader


class TieredLogDataLoader:
    """For use with tiered language models.

    Prepares batches that include several steps.
    """

    def __init__(
        self,
        filepaths,
        tokenizer: Tokenizer,
        task,
        batch_size=100,
        num_steps=5,
        delimiter=" ",
    ):
        self.tokenizer: Tokenizer = tokenizer
        self.task = task
        self.delimiter = delimiter  # delimiter for input file
        self.mb_size = batch_size  # the number of users in a batch
        self.num_steps = num_steps  # The number of log lines for each user in a batch
        self.user_logs: Dict[int, List[dict]] = {}
        self.staggler_num_steps = 1
        # the list of users who are ready to be included in the next batch
        # (i.e. whose # of saved log lines are greater than or equal to the self.num_steps)
        self.batch_ready_users_list: List[int] = []
        self.filepaths = filepaths
        self.using_cuda = Application.instance().using_cuda

        # __iter__ attributes
        # Flush = entire file has been read and a full batch of different users can no longer be produced
        self.flush = False
        # Used by the trainer to signal if the rest of the current file should
        # be skipped when flush is reached
        self.skip_file = False

    def __iter__(self):
        for datafile in self.filepaths:
            self.flush = False
            self.skip_file = False

            # Feed one file at a time to parse_multiple_files so we can keep track of flush more easily
            file_reader = parse_multiple_files([datafile])

            while True:
                batch_data = []
                if self.skip_file:
                    # Skip the rest of the current file, because it is flush and
                    # we're currently training (see train_loop.py)
                    break
                while not batch_data:
                    if not self.flush:
                        try:
                            # Get the next line
                            datadict = prepare_datadict(next(file_reader), self.task, self.tokenizer)
                            user = datadict["user"].item()
                            if user not in self.user_logs:
                                self.user_logs[user] = []
                            self.user_logs[user].append(datadict)
                            if user not in self.batch_ready_users_list and len(self.user_logs[user]) >= self.num_steps:
                                self.batch_ready_users_list.append(user)
                        except StopIteration:
                            # Failed to get line because file is empty - activate flush
                            self.flush = True

                    # Before the data loader reads the last line of the log - we only want fullsized batches (mb_size)
                    if len(self.batch_ready_users_list) >= self.mb_size and not self.flush:
                        batch_data = self.get_batch_data()

                    # When the data loader has read the last line of the log - we accept any size of batch
                    elif len(self.batch_ready_users_list) > 0 and self.flush:
                        batch_data = self.get_batch_data()

                    # Activate the staggler mode - accept batches with smaller number of steps than num_steps
                    elif len(self.batch_ready_users_list) == 0 and self.flush:
                        if self.num_steps == self.staggler_num_steps:
                            break
                        self.mb_size = self.num_steps * self.mb_size
                        self.num_steps = self.staggler_num_steps

                if not batch_data:
                    break

                # Create the batch by collating the datadicts. Also, if jagged, pad
                # the input fields to the length of the longest sequence in the batch
                batch = self.tiered_collate_fn(batch_data)
                # Add the user that each line belongs to to model input
                batch["input"] = (batch["user"][0], batch["input"])

                yield batch

    def tiered_collate_fn(self, data):
        """Pads the input fields to the length of the longest sequence in the
        batch."""
        batch = {}

        # Prep the batch structure (with python lists for simplicity)
        for key in data[0][0]:
            batch[key] = [[] for _ in range(self.num_steps)]

        # Fill the batch structure with the input data
        for sample in data:
            for step, line_sample in enumerate(sample):
                for key in line_sample:
                    batch[key][step].append(line_sample[key])

        # batch is a dict of the batch entries (input, target, mask, user, day, etc.)
        # Each of the sequence entries (input, target, mask)
        # are of shape [num_steps, batchsize, sequence], e.g. [3, 64, sequence]
        # Where sequence varies (if self.jagged=True).

        if self.tokenizer.jagged:
            # First pad within each num_step so that we get a uniform sequence_length within each num_step
            fields_to_pad = ["input", "target"]
            for key in fields_to_pad:
                for step in range(self.num_steps):
                    batch[key][step] = pad_sequence(batch[key][step], batch_first=True, padding_value=0)
            # Next pad across the num_step so that we have a uniform sequence_length across the entire batch
            for key in fields_to_pad:
                # Swap the batchsize and sequence_length dimensions to allow padding
                for step in range(self.num_steps):
                    batch[key][step] = batch[key][step].transpose(0, 1)
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
                # Reverse the transposition of batchsize and sequence_length - batch[key] is now a tensor
                batch[key] = batch[key].transpose(1, 2)

            batch["mask"] = batch["input"] != self.tokenizer.pad_idx

        # Convert the remaining python lists to tensors
        for key, value in batch.items():
            for step in range(self.num_steps):
                if isinstance(value[step], list):
                    value[step] = torch.stack(value[step])
            if isinstance(value, list):
                batch[key] = torch.stack(value)

        return batch

    def split_batch(self, batch: dict):
        """Splits a batch into variables containing relevant data."""

        X = batch["input"]
        Y = batch["target"]

        # Optional fields
        L = batch.get("length")
        M = batch.get("mask")

        if self.using_cuda:
            X = (X[0].cuda(), X[1].cuda())
            Y = Y.cuda()
            if M is not None:
                M = M.cuda()

        split_batch = {
            "X": X,
            "Y": Y,
            "L": L,
            "M": M,
            "user": batch["user"],
            "second": batch["second"],
            "red_flag": batch["red"],
        }

        # Grab evaluation data

        return split_batch

    def get_batch_data(self):
        batch_data = []
        # Loop over users that have enough lines loaded to be used in a batch
        for user in self.batch_ready_users_list[: self.mb_size]:
            # Add user's lines to the batch
            batch_data.append(self.user_logs[user][0 : self.num_steps])
            # Update user's saved lines
            self.user_logs[user] = self.user_logs[user][self.num_steps :]
            # Remove user from list if it now doesn't have enough lines left to be used in another batch
            if len(self.user_logs[user]) < self.num_steps:
                self.batch_ready_users_list.remove(user)
        return batch_data


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
