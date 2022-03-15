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
    """For masking output of language model for jagged sequences for correct
    gradient update. Sequence length of 0 will output nan for that row of mask
    so don't do this.

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


class MapMultilineDataset(LogDataset, Dataset):
    """Provides data via __getitem__, allowing arbitrary data entries to be
    accessed via index.

    Provides sequences of loglines of length window_size * 2 - 1.
    """

    def __init__(self, filepaths, bidirectional, skipsos, jagged, delimiter=" ", window_size=100) -> None:
        super().__init__(filepaths, bidirectional, skipsos, jagged, delimiter)

        self.window_size = window_size

        self.loglines = []
        self.skipsos = True
        self.skipeos = True
        iterator = parse_multiple_files(self.filepaths, jagged, bidirectional, skipsos, raw_lines=True)

        self.loglines.extend(iterator)
        # Length explanation: Divide by window size and floor since we can't/don't want to pass on incomplete sequences
        # Subtract one because we lose the first window_size lines because they can't have a history of length window_size
        self.length = (len(self.loglines) // self.window_size) - 1

    def __getitem__(self, index):
        # Actual input to the model (that will produce an output prediction): window_size
        # Extra history before the start of this input needed to ensure a full window_size history for every entry: window_size-1
        # Length of each item: 2*window_size - 1 long
        start_index = index * self.window_size
        end_index = start_index + 2*self.window_size  # Add 1 line that will be the target for the last input
        sequence = self.loglines[start_index:end_index]
        parsed_sequence = self.parse_lines(sequence)
        return parsed_sequence

    def __len__(self):
        return self.length

    def parse_lines(self, lines):
        datadict = {
            "line": [],
            "second": [],
            "day": [],
            "user": [],
            "red": [],
            "input": [],
            "target": [],
        }

        metadata_offset = 5
        offset = int(self.skipsos)
        input_start = metadata_offset + offset

        this_sequence_len = len(lines)

        for idx, line in enumerate(lines):
            split_line = line.strip().split(self.delimiter)
            split_line = [int(x) for x in split_line]
            data = torch.LongTensor(split_line)

            length = data.shape[0] - metadata_offset - int(self.skipeos) - int(self.skipsos)
            input_end = input_start + length

            # The last line in the input is only used as the target for the 2nd to last line, not as input
            if idx < this_sequence_len - 1:
                datadict["input"].append(data[input_start:input_end])
            # The first window_size lines processed are not the target of anything (in this sequence) - only history
            if idx > self.window_size - 1:
                datadict["line"].append(data[0])
                datadict["second"].append(data[1])
                datadict["day"].append(data[2])
                datadict["user"].append(data[3])
                datadict["red"].append(data[4])
                datadict["target"].append(data[input_start:input_end])

        return datadict


class IterableUserMultilineDataset(LogDataset, IterableDataset):
    """Provides data via __iter__, allowing data to be accessed in order
    only.

    Provides sequences of loglines of length window_size * 2 - 1. Each sequence contains loglines from a single user.
    """

    def __init__(self, filepaths, bidirectional, skipsos, jagged, delimiter=" ", window_size=100) -> None:
        super().__init__(filepaths, bidirectional, skipsos, jagged, delimiter)

        self.window_size = window_size

        self.loglines = []
        self.skipsos = True
        self.skipeos = True
        self.data = parse_multiple_files(self.filepaths, jagged, bidirectional, skipsos, raw_lines=True)
        self.user_loglines = {}

    def __iter__(self):
        # Actual input to the model (that will produce an output prediction): window_size
        # Extra history before the start of this input needed to ensure a full window_size history for every entry: window_size-1
        # Length of each item: 2*window_size - 1 long
        for line in self.data:
            line_data = self.parse_line(line)
            line_user = line_data["user"].item()
            if line_user not in self.user_loglines:
                self.user_loglines[line_user] = []
            self.user_loglines[line_user].append(line_data)
            # Check if this user has enough lines to produce a sequence:
            # window_size*2 (window_size-1 history, window_size inputs, 1 final target)
            if len(self.user_loglines[line_user]) >= self.window_size * 2:
                yield self.produce_output_sequence(line_user)

    def parse_line(self, line):
        datadict = {
            "line": [],
            "second": [],
            "day": [],
            "user": [],
            "red": [],
            "data": [],
        }

        metadata_offset = 5
        offset = int(self.skipsos)
        input_start = metadata_offset + offset

        split_line = line.strip().split(self.delimiter)
        split_line = [int(x) for x in split_line]
        data = torch.LongTensor(split_line)

        length = data.shape[0] - metadata_offset - int(self.skipeos) - int(self.skipsos)
        input_end = input_start + length

        datadict["line"] = data[0]
        datadict["second"] = data[1]
        datadict["day"] = data[2]
        datadict["user"] = data[3]
        datadict["red"] = data[4]
        datadict["data"] = data[input_start:input_end]

        return datadict

    def produce_output_sequence(self, user):
        """Puts together a sequence of loglines from a single user from the data that's been read in so far."""
        datadict = {
            "line": [],
            "second": [],
            "day": [],
            "user": [],
            "red": [],
            "input": [],
            "target": [],
        }

        lines = self.user_loglines[user]

        this_sequence_len = len(lines)

        for idx, line_data in enumerate(lines):
            # The last line in the input is only used as the target for the 2nd to last line, not as input
            if idx < this_sequence_len - 1:
                datadict["input"].append(line_data["data"])
            # The first window_size lines processed are not the target of anything (in this sequence) - only history
            if idx > self.window_size - 1:
                datadict["line"].append(line_data["line"])
                datadict["second"].append(line_data["second"])
                datadict["day"].append(line_data["day"])
                datadict["user"].append(line_data["user"])
                datadict["red"].append(line_data["red"])
                datadict["target"].append(line_data["data"])
        # Remove all lines from this user, except the ones necessary for history for the next sequence (last window_size - 1)
        lines = lines[self.window_size - 1:]
        self.user_loglines[user] = lines
        return datadict


class LogDataLoader(DataLoader):
    """Wrapper class around torch's DataLoader, used for non-tiered data
    loading.

    Provides a function to split the batch provided by the data loader.
    """

    def __init__(self, dataset, batch_size, shuffle, collate_fn=None, batch_sampler=None):
        if batch_sampler is not None:
            super().__init__(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler)
        else:
            super().__init__(
                dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
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
        }

        # Grab evaluation data
        split_batch["user"] = batch["user"]
        split_batch["second"] = batch["second"]
        split_batch["red_flag"] = batch["red"]

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
):
    def create_tiered_data_loader(filepath):
        data_handler = TieredLogDataLoader(
            filepath,
            sentence_length,
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


def create_data_loaders_multiline(
    filepath, batch_size, bidir, skipsos, jagged, window_size, memory_type, shuffle=False, dataset_split=None
):
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
            for sequence_index in range(len(batch[key])):
                if isinstance(batch[key][sequence_index], list):
                    batch[key][sequence_index] = torch.stack(batch[key][sequence_index])
            if isinstance(batch[key], list):
                batch[key] = torch.stack(batch[key])

        return batch
    
    if memory_type.lower() == "global":
        dataset = MapMultilineDataset(filepath, bidir, skipsos, jagged, window_size=window_size)
    elif memory_type.lower() == "user":
        dataset = IterableUserMultilineDataset(filepath, bidir, skipsos, jagged, window_size=window_size)
    else:
        raise ValueError(f"Invalid memory_type. Expected 'global' or 'user', got '{memory_type}'")

    # Split the dataset according to the split list
    if dataset_split is not None and not isinstance(dataset, IterableDataset):
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


def load_data_multiline(
    data_folder,
    train_files,
    test_files,
    batch_size,
    bidir,
    skipsos,
    jagged,
    window_size,
    memory_type,
    train_val_split=[1, 0],
):
    filepaths_train = [path.join(data_folder, f) for f in train_files]
    filepaths_eval = [path.join(data_folder, f) for f in test_files]
    train_loader, val_loader = create_data_loaders_multiline(
        filepaths_train,
        batch_size,
        bidir,
        skipsos,
        jagged,
        window_size=window_size,
        memory_type=memory_type,
        dataset_split=train_val_split,
    )
    test_loader, _ = create_data_loaders_multiline(
        filepaths_eval, batch_size, bidir, skipsos, jagged, window_size=window_size, memory_type=memory_type, dataset_split=None
    )

    return train_loader, val_loader, test_loader


class TieredLogDataLoader:
    """For use with tiered language models.

    Prepares batches that include several steps.
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
    ):
        self.sentence_length = sentence_length
        self.jagged = jagged
        self.skipsos = skipsos
        self.bidir = bidir
        self.delimiter = delimiter  # delimiter for input file
        self.mb_size = batch_size  # the number of users in a batch
        self.num_steps = num_steps  # The number of log lines for each user in a batch
        self.user_logs = {}
        self.staggler_num_steps = 1
        # the list of users who are ready to be included in the next batch
        # (i.e. whose # of saved log lines are greater than or equal to the self.num_steps)
        self.batch_ready_users_list = []
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
            file_reader = parse_multiple_files([datafile], self.jagged, self.bidir, self.skipsos)

            while True:
                batch_data = []
                if self.skip_file:
                    # Skip the rest of the current file, because it is flush and
                    # we're currently training (see train_loop.py)
                    break
                while batch_data == []:
                    if not self.flush:
                        try:
                            # Get the next line
                            datadict = next(file_reader)
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

                if batch_data == []:
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
        # Each of the sequence entries (input, target, mask) are of shape [num_steps, batchsize, sequence], e.g. [3, 64, sequence]
        # Where sequence varies (if self.jagged=True).

        if self.jagged:
            # First pad within each num_step so that we get a uniform sequence_length within each num_step
            fields_to_pad = ["input", "target", "mask"]
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
        # Convert the remaining python lists to tensors
        for key in batch:
            for step in range(self.num_steps):
                if isinstance(batch[key][step], list):
                    batch[key][step] = torch.stack(batch[key][step])
            if isinstance(batch[key], list):
                batch[key] = torch.stack(batch[key])

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
        }

        # Grab evaluation data
        split_batch["user"] = batch["user"]
        split_batch["second"] = batch["second"]
        split_batch["red_flag"] = batch["red"]

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
