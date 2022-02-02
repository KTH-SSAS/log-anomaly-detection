import torch
from torch.utils.data import Dataset, IterableDataset

from log_analyzer.tokenizer.tokenizer_neo import LANLTokenizer

AUTOREGRESSIVE_LM = 0
BIDIR_LSTM_LM = 1
MASKED_LM = 2

SECONDS_PER_DAY = 86400


def prepare_datadict(line, task, tokenizer: LANLTokenizer):

    fields = line.strip().split(",")

    # Remove timestamp and red team flag from input
    to_tokenize = fields[1:-1]

    tokenized_line = tokenizer.tokenize(to_tokenize)

    datadict = {
        "second": torch.LongTensor([int(fields[0])]),
        "day": torch.LongTensor([int(fields[0]) // SECONDS_PER_DAY]),
        "red": torch.BoolTensor([int(fields[-1])]),
        "user": torch.LongTensor([tokenizer.vocab.token2idx(fields[2], "src_user")]),
    }

    if task == AUTOREGRESSIVE_LM:
        # data_in = tokenized_line[:-1]
        # label = tokenized_line[1:]
        pass
    elif task == BIDIR_LSTM_LM:
        # data_in = tokenized_line
        # label = tokenized_line[1:-1]
        pass
    elif task == MASKED_LM:
        data_in, label, _ = tokenizer.mask_tokens(tokenized_line)
    else:
        raise RuntimeError("Invalid Task")

    datadict["input"] = torch.LongTensor(data_in)
    datadict["label"] = torch.LongTensor(label)

    return datadict


class LogDataset:
    def __init__(self, filepaths, tokenizer, task) -> None:
        super().__init__()

        if not isinstance(filepaths, list):
            self.filepaths = [filepaths]
        else:
            self.filepaths = filepaths

        self.task = task
        self.tokenizer = tokenizer
        self.filepaths = filepaths


class LANLDataset(Dataset, LogDataset):
    def __init__(self, filepaths, tokenizer, task) -> None:
        super().__init__(filepaths, tokenizer, task)

    def __getitem__(self, index):

        pass


class IterableLANLDataset(IterableDataset, LogDataset):
    def __init__(self, filepaths, tokenizer, task) -> None:
        super().__init__(filepaths, tokenizer, task)

    def __iter__(self):
        for filename in self.filepaths:
            with open(filename, "r") as f:
                for line in f:
                    datadict = prepare_datadict(line, task=self.task, tokenizer=self.tokenizer)
                    yield datadict
