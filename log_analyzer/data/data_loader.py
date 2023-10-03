"""Data loading functions."""

from functools import partial
from os import path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

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


def prepare_datadict(line: str, task: str, tokenizer: Tokenizer, test: bool = False) -> dict:

    fields = line.strip().split(",")
    second = int(fields[0])
    user = tokenizer.user_idx(fields[1])

    add_sos, add_eos = tokens_to_add(task)

    if tokenizer.include_timestamp:
        # Remove red team flag from input
        to_tokenize = line[: -3]
    else:
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
        data_in, label = tokenizer.mask_tokens(tokenized_line, test=test)
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

    def __init__(self, filepaths: Union[str, List[str]], tokenizer: Tokenizer, task: str, test: bool = False) -> None:
        self.tokenizer: Tokenizer = tokenizer
        self.task = task
        self.test = test

        if isinstance(filepaths, str):
            filepaths = [filepaths]
        self.filepaths = filepaths


class MapLogDataset(LogDataset, Dataset):
    """Provides data via __getitem__, allowing arbitrary data entries to be
    accessed via index."""

    def __init__(self, filepaths, tokenizer, task, test=False) -> None:
        super().__init__(filepaths, tokenizer, task, test)
        self.log_lines = []
        iterator = parse_multiple_files(self.filepaths)
        self.log_lines.extend(iterator)

    def __getitem__(self, index):
        log_line = self.log_lines[index]
        parsed_line = prepare_datadict(log_line, self.task, self.tokenizer, test=self.test)
        return parsed_line

    def __len__(self):
        return len(self.log_lines)


class IterableLogDataset(LogDataset, IterableDataset):  # pylint: disable=abstract-method
    """Provides data via __iter__, allowing data to be accessed in order
    only."""

    def __init__(self, filepaths, tokenizer, task, test=False) -> None:
        super().__init__(filepaths, tokenizer, task, test)
        self.refresh_iterator()

    def __iter__(self):
        return self.iterator

    def refresh_iterator(self):
        def generate_iterator():
            for line in parse_multiple_files(self.filepaths):
                yield prepare_datadict(line, self.task, self.tokenizer, test=self.test)

        self.iterator = generate_iterator()


class MultilineLogDataset:
    """Virtual superclass for multiline datasets."""

    def __init__(self, task, shift_window: int = 100, batch_entry_size: Optional[int] = None):
        assert task == SENTENCE_LM, f"Task must be 'sentence-lm' when using this dataset. Got '{task}'."

        self.shift_window = shift_window
        # batch_entry_size defines how many lines (in addition to context) we send per batch entry
        # default is shift_window (with shift_window - 1 context lines)
        self.batch_entry_size = batch_entry_size if batch_entry_size is not None else shift_window
        self.skipsos = True
        self.skipeos = True

    def produce_output_sequence(self, log_lines, context_lines):
        """Puts together a sequence of loglines from a single user from the
        data that's been read in so far.

        If the user does not have enough log lines (self.batch_entry_size) we pad by appending padding (0).

        If the user does not have any context lines, we haven't sent any sequence from this user yet.
        We therefore pad the context with 0s (shift_window), then append padding to total length
        shift_window + batch_entry-size - 1.
        """
        num_inputs = self.shift_window + self.batch_entry_size - 1
        num_targets = self.batch_entry_size

        datadict: Dict[str, torch.Tensor] = {
            "second": torch.zeros((num_targets), dtype=torch.long),
            "day": torch.zeros((num_targets), dtype=torch.long),
            "user": torch.zeros((num_targets), dtype=torch.long),
            "red": torch.zeros((num_targets), dtype=torch.long),
            "input": torch.zeros((num_inputs, log_lines[0]["input"].shape[0]), dtype=torch.long),
            "target": torch.zeros((num_targets, log_lines[0]["input"].shape[0]), dtype=torch.long),
            "length": torch.zeros((num_targets), dtype=torch.long),
        }

        # First add the context lines
        for idx, line_data in enumerate(context_lines, start=self.shift_window - len(context_lines)):
            datadict["input"][idx] = line_data["input"]

        for idx, line_data in enumerate(log_lines):
            # The last line in the input is only used as the target for the 2nd to last line, not as input
            if idx < len(log_lines) - 1:
                datadict["input"][idx + self.shift_window] = line_data["input"]
            datadict["second"][idx] = line_data["second"]
            datadict["day"][idx] = line_data["day"]
            datadict["user"][idx] = line_data["user"]
            datadict["red"][idx] = line_data["red"]
            datadict["target"][idx] = line_data["input"]  # A line's target is the same as the next line's input
            datadict["length"][idx] = line_data["length"]

        return datadict


class IterableUserMultilineDataset(IterableLogDataset, MultilineLogDataset):
    """Provides data via __iter__, allowing data to be accessed in order only.

    Provides sequences of loglines of length shift_window * 2 - 1. Each sequence contains loglines from a single user.
    """

    def __init__(self, filepaths, tokenizer, task, shift_window=100, batch_entry_size=None) -> None:
        super().__init__(filepaths, tokenizer, task)
        MultilineLogDataset.__init__(self, task, shift_window, batch_entry_size)

        self.training = False  # Assume we're not a training dataset.
        self.data = list(parse_multiple_files(self.filepaths))
        # Stores the newest lines from each user that have not yet been batched
        self.user_loglines: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        # Stores the last shift_window lines from each user that have been batched (for context)
        self.user_context: Dict[str, List[Dict[str, torch.Tensor]]] = {}

        self.refresh_iterator()

    def __iter__(self):
        return self.iterator

    def __getitem__(self, index):
        raise NotImplementedError("Iterable dataset must be accessed via __iter__.")

    def refresh_iterator(self):
        """Generates a (new) iterator over the data as specified by class
        parameters."""

        def generate_iterator():
            # For each user, we split its sequence of log lines into buckets that are shift_window long.
            # For each batch entry we need 2 buckets: 1 that will be predicted ("classified") and the bucket before it
            # to provide full context (i.e. shift_window lines of input) for each log line.
            for line in self.data:
                line_data = prepare_datadict(line, self.task, self.tokenizer)
                line_user = line_data["user"].item()
                if line_user not in self.user_loglines:
                    self.user_loglines[line_user] = []
                    # For the first bucket of each user, the only context is a border-padded line
                    # (that user's first line, repeated as padding)
                    self.user_context[line_user] = [line_data]
                self.user_loglines[line_user].append(line_data)
                # Check if this user has a full bucket of log lines.
                if len(self.user_loglines[line_user]) == self.batch_entry_size:
                    yield self.get_output(line_user, self.user_loglines[line_user], self.user_context[line_user])
            # When we've exhausted the data, return the incomplete sequences (padded up to full length)
            for line_user, user_lines in self.user_loglines.items():
                if len(user_lines):
                    yield self.get_output(line_user, user_lines, self.user_context[line_user])

        # Clear the leftover lines currently stored
        self.user_loglines = {}
        self.user_context = {}
        self.iterator = generate_iterator()

    def get_output(self, user, log_lines, context_lines):
        sequence = self.produce_output_sequence(log_lines, context_lines)
        self.update_user_context(user)
        return sequence

    def update_user_context(self, user):
        # Update this user's context - take the last self.shift_window lines from its saved lines
        self.user_context[user].extend(self.user_loglines[user])
        self.user_context[user] = self.user_context[user][-self.shift_window :]
        # Remove all lines from the user's line list
        self.user_loglines[user] = []


class MapMultilineDataset(MapLogDataset, MultilineLogDataset):
    """Provides data via __getitem__, allowing arbitrary data entries to be accessed via index in arbitrary order.

    Provides sequences of loglines of length shift_window * 2 - 1. Each sequence contains loglines from a single user.
    """

    def __init__(
        self,
        filepaths,
        tokenizer: Tokenizer,
        task,
        memory_type: str = "user",
        shift_window: int = 100,
        batch_entry_size: Optional[int] = None,
    ) -> None:
        assert task == SENTENCE_LM, f"Task must be 'sentence-lm' when using this dataset. Got '{task}'."
        super().__init__(filepaths, tokenizer, task)
        MultilineLogDataset.__init__(self, task, shift_window, batch_entry_size)

        self.shift_window = shift_window
        # batch_entry_size defines how many lines (in addition to context) we send per batch entry
        # default is shift_window (with shift_window - 1 context lines)
        self.batch_entry_size = batch_entry_size if batch_entry_size is not None else shift_window
        self.skipsos = True
        self.skipeos = True

        memory_type = memory_type.lower()
        if memory_type not in ("user", "global"):
            raise ValueError(f"Memory type must be 'user' or 'global'. Got '{memory_type}'.")

        self.data: Dict[int, List[str]] = {}
        self.items: List[Tuple[int, int]] = []
        for rawline in parse_multiple_files(filepaths):
            user = tokenizer.user_idx(rawline.split(",", maxsplit=2)[1]) if memory_type == "user" else 0
            if user not in self.data:
                self.data[user] = []
            self.data[user].append(rawline)
            # If this line is the start of a new batch_entry, add this user (and index) to the list of items
            if len(self.data[user]) % self.batch_entry_size == 1:
                self.items.append((user, int(len(self.data[user]) // self.batch_entry_size)))

        for user, lines in self.data.items():
            # Add the padding/sos line to the start of each user's sequence
            self.data[user].insert(0, lines[0])  # pylint: disable=unnecessary-dict-index-lookup

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        user, batch_entry_index = self.items[index]
        # user: the user for this batch entry
        # batch_entry_index: the index of this batch entry
        # line_index: the index of the first main_line (i.e. not context) to appear in this batch entry
        line_index = batch_entry_index * self.batch_entry_size

        make_dict = partial(prepare_datadict, task=self.task, tokenizer=self.tokenizer)

        context_lines_end_index = line_index + 1
        context_lines_start_index = max(0, context_lines_end_index - self.shift_window)
        context_lines = self.data[user][context_lines_start_index:context_lines_end_index]
        context_lines = list(map(make_dict, context_lines))
        # The first line in each user's list is a padding line akin to CNN image border padding (repeat the first line)
        # This is ignored for the main_lines, thus shifted by 1
        main_lines_start_index = line_index + 1
        main_lines_end_index = min(len(self.data[user]), main_lines_start_index + self.batch_entry_size)
        main_lines = self.data[user][main_lines_start_index:main_lines_end_index]
        main_lines = list(map(make_dict, main_lines))

        return self.produce_output_sequence(main_lines, context_lines)


class LogDataLoader(DataLoader):
    """Wrapper class around torch's DataLoader, used for non-tiered data
    loading.

    Provides a function to split the batch provided by the data loader.
    """

    def __init__(self, dataset, batch_size, shuffle, collate_function):

        num_workers = 7 if isinstance(dataset, MapLogDataset) else 0

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_function,
            num_workers=num_workers,
            pin_memory=True,
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
        return split_batch


class MultilineDataLoader(LogDataLoader):
    """Dataloader for multiline datasets and models.

    Provides a function to split the batch provided by the data loader (including filtering out entries that are too
    masked to be properly processed - i.e. don't contain full context for in line sequence).
    """

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
        if "target_mask" in batch:
            split_batch["target_mask"] = batch["target_mask"]
        return split_batch


def load_data_tiered(
    data_folder,
    train_files,
    validation_files,
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
    filepaths_valid = [path.join(data_folder, f) for f in validation_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_tiered_data_loader(filepaths_train, batch_sizes[0])
    if len(validation_files) > 0:
        val_loader = create_tiered_data_loader(filepaths_valid, batch_sizes[1])
    else:
        val_loader = None
    test_loader = create_tiered_data_loader(filepaths_eval, batch_sizes[1])
    return train_loader, val_loader, test_loader


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


def create_data_loader(
    filepaths: List[str],
    batch_size: int,
    tokenizer: Tokenizer,
    task: str,
    shuffle: bool = False,
    test: bool = False,
) -> LogDataLoader:
    """Creates and returns a data loader."""

    dataset: LogDataset
    dataset = MapLogDataset(filepaths, tokenizer, task, test=test)
    # else:
    #     dataset = IterableLogDataset(filepaths, tokenizer, task, test=test)

    collate = partial(collate_fn, jagged=tokenizer.jagged)
    data_handler = LogDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_function=collate)

    return data_handler

def multiline_collate_fn(data, pad_idx=0):
    """Pads the input fields to the length of the longest sequence in the
    batch."""
    batch = {}

    for key in data[0]:
        batch[key] = []

    for sample in data:
        for key in sample:
            batch[key].append(sample[key])

    for key, value in batch.items():
        for index, sequence in enumerate(value):
            if isinstance(sequence, list):
                value[index] = torch.stack(sequence)
        if isinstance(value, list):
            batch[key] = torch.stack(value)

    # Add the input padding mask - pad_idx is 0
    batch["mask"] = torch.all(batch["input"] != pad_idx, dim=2)

    return batch

def create_data_loader_multiline(
    filepaths: List[str],
    batch_size: int,
    tokenizer: Tokenizer,
    task: str,
    shift_window: int,
    memory_type: str,
    shuffle: bool = False,
) -> MultilineDataLoader:
    """Creates and returns a data loader."""
    dataset = MapMultilineDataset(filepaths, tokenizer, task, memory_type=memory_type, shift_window=shift_window)
    collate = partial(multiline_collate_fn, pad_idx=0)
    data_handler = MultilineDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_function=collate)
    return data_handler


def load_data(
    data_folder: str,
    train_files: List[str],
    validation_files: List[str],
    test_files: List[str],
    batch_sizes: Tuple[int, int],
    tokenizer: Tokenizer,
    task: str,
    shuffle_train_data: bool = True,
):
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_valid = [path.join(data_folder, f) for f in validation_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_data_loader(filepaths_train, batch_sizes[0], tokenizer, task, shuffle_train_data)
    if len(validation_files) > 0:
        val_loader = create_data_loader(filepaths_valid, batch_sizes[1], tokenizer, task, shuffle=False)
    else:
        val_loader = None
    test_loader = create_data_loader(filepaths_eval, batch_sizes[1], tokenizer, task, shuffle=False, test=True)
    return train_loader, val_loader, test_loader


def load_data_multiline(
    data_folder: str,
    train_files: List[str],
    validation_files: List[str],
    test_files: List[str],
    batch_sizes: Tuple[int, int],
    tokenizer: Tokenizer,
    task: str,
    shift_window: int,
    memory_type: str,
    shuffle_train_data: bool = True,
):
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_valid = [path.join(data_folder, f) for f in validation_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader = create_data_loader_multiline(
        filepaths_train,
        batch_sizes[0],
        tokenizer,
        task,
        shift_window=shift_window,
        memory_type=memory_type,
        shuffle=shuffle_train_data,
    )
    if len(validation_files) > 0:
        val_loader = create_data_loader_multiline(
            filepaths_valid,
            batch_sizes[1],
            tokenizer,
            task,
            shift_window=shift_window,
            memory_type=memory_type,
        )
    else:
        val_loader = None
    test_loader = create_data_loader_multiline(
        filepaths_eval,
        batch_sizes[1],
        tokenizer,
        task,
        shift_window=shift_window,
        memory_type=memory_type,
    )
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
        self.batch_size = batch_size  # the number of users in a batch
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
                    if len(self.batch_ready_users_list) >= self.batch_size and not self.flush:
                        batch_data = self.get_batch_data()

                    # When the data loader has read the last line of the log - we accept any size of batch
                    elif len(self.batch_ready_users_list) > 0 and self.flush:
                        batch_data = self.get_batch_data()

                    # Activate the staggler mode - accept batches with smaller number of steps than num_steps
                    elif len(self.batch_ready_users_list) == 0 and self.flush:
                        if self.num_steps == self.staggler_num_steps:
                            break
                        self.batch_size = self.num_steps * self.batch_size
                        self.num_steps = self.staggler_num_steps
                        # Update batch_ready_users_list for the smaller num_steps
                        for user, logs in self.user_logs.items():
                            if len(logs) >= self.num_steps:
                                self.batch_ready_users_list.append(user)

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
        for user in self.batch_ready_users_list[: self.batch_size]:
            # Add user's lines to the batch
            batch_data.append(self.user_logs[user][0 : self.num_steps])
            # Update user's saved lines
            self.user_logs[user] = self.user_logs[user][self.num_steps :]
            # Remove user from list if it now doesn't have enough lines left to be used in another batch
            if len(self.user_logs[user]) < self.num_steps:
                self.batch_ready_users_list.remove(user)
        return batch_data
