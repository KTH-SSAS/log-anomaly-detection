"""Data loading functions."""

from functools import partial
from os import path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, random_split

from log_analyzer.application import Application
from log_analyzer.tokenizer.tokenizer_neo import Tokenizer

AUTOREGRESSIVE_LM = "lm"
BIDIR_LSTM_LM = "bidir-lm"
MASKED_LM = "masked-lm"
SENTENCE_LM = "sentence-lm"

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


def tokens_to_add(task: str) -> Tuple[bool, bool]:
    add_sos: bool
    add_eos: bool
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
    elif task == SENTENCE_LM:
        # Include all tokens in both input and target
        data_in = tokenized_line
        label = tokenized_line
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

    def __init__(self, filepaths: Union[str, List[str]], tokenizer: Tokenizer, task: str) -> None:
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

        self.log_lines = []
        iterator = parse_multiple_files(self.filepaths)
        self.log_lines.extend(iterator)

    def __getitem__(self, index):
        log_line = self.log_lines[index]
        parsed_line = prepare_datadict(log_line, self.task, self.tokenizer)
        return parsed_line

    def __len__(self):
        return len(self.log_lines)


class IterableLogDataset(LogDataset, IterableDataset):  # pylint: disable=abstract-method
    """Provides data via __iter__, allowing data to be accessed in order
    only."""

    def __init__(self, filepaths, tokenizer, task) -> None:
        super().__init__(filepaths, tokenizer, task)
        self.refresh_iterator()

    def __iter__(self):
        return self.iterator

    def refresh_iterator(self):
        def generate_iterator():
            for line in parse_multiple_files(self.filepaths):
                yield prepare_datadict(line, self.task, self.tokenizer)

        self.iterator = generate_iterator()


class MapMultilineDataset(LogDataset, Dataset):
    """Provides data via __getitem__, allowing arbitrary data entries to be
    accessed via index.

    Provides sequences of loglines of length shift_window * 2 - 1.
    """

    def __init__(self, filepaths, tokenizer, task, shift_window=100) -> None:
        assert task == SENTENCE_LM, "Task must be 'sentence-lm' when using this dataset."
        super().__init__(filepaths, tokenizer, task)

        self.shift_window = shift_window

        self.loglines = []
        self.skipsos = True
        self.skipeos = True
        iterator = parse_multiple_files(self.filepaths)

        self.loglines.extend(iterator)
        # Length explanation: Divide by window size and floor since we can't/don't want to pass on incomplete sequences
        # -1 because we lose the first shift_window lines because they can't have a history of length shift_window
        self.length = (len(self.loglines) // self.shift_window) - 1

    def __getitem__(self, index):
        # Actual input to the model (that will produce an output prediction): shift_window
        # Extra history before needed to ensure a full shift_window history for every entry: shift_window-1
        # Length of each item: 2*shift_window - 1 long
        start_index = index * self.shift_window
        end_index = start_index + 2 * self.shift_window  # Add 1 line that will be the target for the last input
        sequence = self.loglines[start_index:end_index]
        parsed_sequence = self.parse_lines(sequence)
        return parsed_sequence

    def __len__(self):
        return self.length

    def parse_lines(self, lines):
        datadict = {
            "second": [],
            "day": [],
            "user": [],
            "red": [],
            "input": [],
            "target": [],
            "length": [],
        }

        this_sequence_len = len(lines)

        for idx, line in enumerate(lines):
            data = prepare_datadict(line, self.task, self.tokenizer)

            # The last line in the input is only used as the target for the 2nd to last line, not as input
            if idx < this_sequence_len - 1:
                datadict["input"].append(data["input"])
            # The first shift_window lines processed are not the target of anything (in this sequence) - only history
            if idx > self.shift_window - 1:
                datadict["second"].append(data["second"])
                datadict["day"].append(data["day"])
                datadict["user"].append(data["user"])
                datadict["red"].append(data["red"])
                datadict["target"].append(data["target"])
                datadict["length"].append(data["length"])

        return datadict


class IterableUserMultilineDataset(LogDataset, IterableDataset):
    """Provides data via __iter__, allowing data to be accessed in order only.

    Provides sequences of loglines of length shift_window * 2 - 1. Each sequence contains loglines from a single user.
    """

    def __init__(self, filepaths, tokenizer, task, shift_window=100) -> None:
        assert task == SENTENCE_LM, f"Task must be 'sentence-lm' when using this dataset. Got '{task}'."
        super().__init__(filepaths, tokenizer, task)

        self.shift_window = shift_window

        self.skipsos = True
        self.skipeos = True
        self.user_loglines: Dict[str, Dict[str, List]] = {}
        self.refresh_iterator()

    def __iter__(self):
        return self.iterator

    def __getitem__(self, index):
        raise NotImplementedError("Iterable dataset must be accessed via __iter__.")

    def refresh_iterator(self):
        """Generates a (new) iterator over the data as specified by class
        parameters."""

        def generate_iterator():
            data = parse_multiple_files(self.filepaths)
            # Actual input to the model (that will produce an output prediction): shift_window
            # Extra history needed to ensure a full shift_window history for every entry: shift_window-1
            # Length of each item: 2*shift_window - 1 long
            for line in data:
                line_data = prepare_datadict(line, self.task, self.tokenizer)
                line_user = line_data["user"].item()
                if line_user not in self.user_loglines:
                    self.user_loglines[line_user] = []
                self.user_loglines[line_user].append(line_data)
                # Check if this user has enough lines to produce a sequence:
                # shift_window*2 (shift_window-1 history, shift_window inputs, 1 final target)
                if len(self.user_loglines[line_user]) >= self.shift_window * 2:
                    yield self.produce_output_sequence(line_user)

        self.iterator = generate_iterator()

    def produce_output_sequence(self, user):
        """Puts together a sequence of loglines from a single user from the
        data that's been read in so far."""
        datadict = {
            "second": [],
            "day": [],
            "user": [],
            "red": [],
            "input": [],
            "target": [],
            "length": [],
        }

        lines = self.user_loglines[user]

        this_sequence_len = len(lines)

        for idx, line_data in enumerate(lines):
            # The last line in the input is only used as the target for the 2nd to last line, not as input
            if idx < this_sequence_len - 1:
                datadict["input"].append(line_data["input"])
            # The first shift_window lines processed are not the target of anything (in this sequence) - only history
            if idx > self.shift_window - 1:
                datadict["second"].append(line_data["second"])
                datadict["day"].append(line_data["day"])
                datadict["user"].append(line_data["user"])
                datadict["red"].append(line_data["red"])
                datadict["target"].append(line_data["target"])
                datadict["length"].append(line_data["length"])
        # Remove all lines from this user not needed for history for the next sequence (last shift_window - 1)
        lines = lines[self.shift_window - 1 :]
        self.user_loglines[user] = lines
        return datadict


class LogDataLoader(DataLoader):
    """Wrapper class around torch's DataLoader, used for non-tiered data
    loading.

    Provides a function to split the batch provided by the data loader.
    """

    def __init__(self, dataset, batch_size, shuffle, collate_function):

        num_workers = 7 if isinstance(dataset, MapLogDataset) else 0

        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_function, num_workers=num_workers
        )
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
    batch_sizes: Tuple[int, int],
    tokenizer: Tokenizer,
    task,
    num_steps,
):
    def create_tiered_data_loader(filepath, batch_size):
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
    train_loader = create_tiered_data_loader(filepaths_train, batch_sizes[0])
    test_loader = create_tiered_data_loader(filepaths_eval, batch_sizes[1])
    return train_loader, test_loader


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


def create_data_loaders(
    filepaths: List[str],
    batch_sizes: Tuple[int, int],
    tokenizer: Tokenizer,
    task: str,
    shuffle: bool = False,
    validation_portion: float = 0,
) -> Sequence[Optional[LogDataLoader]]:
    """Creates and returns 2 data loaders.

    If dataset_split is not provided the second data loader is instead
    set to None.
    """

    dataset: LogDataset
    if shuffle or validation_portion > 0:
        dataset = MapLogDataset(filepaths, tokenizer, task)
    else:
        dataset = IterableLogDataset(filepaths, tokenizer, task)

    datasets: Sequence[Union[Subset, LogDataset]]
    # Split the dataset according to the split list
    if 0 < validation_portion < 1 and isinstance(dataset, MapLogDataset):
        # Convert val portion into length as proportion of dataset length
        val_size = int(validation_portion * len(dataset))
        train_size = len(dataset) - val_size

        datasets = random_split(dataset, (train_size, val_size))
    else:
        # Return just a single dataset
        datasets = [dataset]

    collate = partial(collate_fn, jagged=tokenizer.jagged)
    data_handlers = [
        LogDataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_function=collate)
        for bs, dataset in zip(batch_sizes, datasets)
    ]

    return data_handlers


def create_data_loaders_multiline(
    filepaths: List[str],
    batch_sizes: Tuple[int, int],
    tokenizer: Tokenizer,
    task: str,
    shift_window: int,
    memory_type: str,
    shuffle: bool = False,
    validation_portion: float = 0,
) -> List[LogDataLoader]:
    """Creates and returns 2 data loaders.

    If dataset_split is not provided the second data loader is instead
    set to None.
    """

    def multiline_collate_fn(data, jagged=False, pad_idx=0):
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
            for index, sequence in enumerate(value):
                if isinstance(sequence, list):
                    value[index] = torch.stack(sequence)
            if isinstance(value, list):
                batch[key] = torch.stack(value)

        return batch

    dataset: LogDataset
    if memory_type.lower() == "global":
        dataset = MapMultilineDataset(filepaths, tokenizer, task, shift_window=shift_window)
    elif memory_type.lower() == "user":
        dataset = IterableUserMultilineDataset(filepaths, tokenizer, task, shift_window=shift_window)
    else:
        raise ValueError(f"Invalid memory_type. Expected 'global' or 'user', got '{memory_type}'")

    datasets: Sequence[Union[Subset, LogDataset]]
    # Split the dataset according to the split list
    if 0 < validation_portion < 1 and isinstance(dataset, MapMultilineDataset):
        # Convert val portion into length as proportion of dataset length
        val_size = int(validation_portion * len(dataset))
        train_size = len(dataset) - val_size

        datasets = random_split(dataset, (train_size, val_size))
    else:
        # Return just a single dataset
        datasets = [dataset]

    collate = partial(multiline_collate_fn, jagged=tokenizer.jagged)

    data_handlers = [
        LogDataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_function=collate)
        for bs, dataset in zip(batch_sizes, datasets)
    ]
    return data_handlers


def load_data(
    data_folder: str,
    train_files: List[str],
    test_files: List[str],
    batch_sizes: Tuple[int, int],
    tokenizer: Tokenizer,
    task: str,
    validation_portion: float = 0,
    shuffle_train_data: bool = True,
):
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader, val_loader = create_data_loaders(
        filepaths_train, batch_sizes, tokenizer, task, shuffle_train_data, validation_portion
    )
    test_loader = create_data_loaders(
        filepaths_eval,
        (batch_sizes[1], batch_sizes[1]),
        tokenizer,
        task,
        shuffle=False,
    )[0]

    return train_loader, val_loader, test_loader


def load_data_multiline(
    data_folder: str,
    train_files: List[str],
    test_files: List[str],
    batch_sizes: Tuple[int, int],
    tokenizer: Tokenizer,
    task: str,
    shift_window: int,
    memory_type: str,
    validation_portion: float = 0,
):
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader, val_loader = create_data_loaders_multiline(
        filepaths_train,
        batch_sizes,
        tokenizer,
        task,
        shift_window=shift_window,
        memory_type=memory_type,
        validation_portion=validation_portion,
    )
    test_loader = create_data_loaders_multiline(
        filepaths_eval,
        (batch_sizes[1], batch_sizes[1]),
        tokenizer,
        task,
        shift_window=shift_window,
        memory_type=memory_type,
    )[0]

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
        self.dataset = None  # This dataloader handles the data directly
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
