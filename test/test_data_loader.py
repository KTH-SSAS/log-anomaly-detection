import pytest
import torch

from log_analyzer.data.data_loader import create_data_loaders
from log_analyzer.tokenizer.tokenizer_neo import CharTokenizer, FieldTokenizer, LANLTokenizer, LANLVocab
from log_analyzer.tokenizer.vocab import GlobalVocab


def batch_equal(v1: torch.Tensor, v2: torch.Tensor):
    assert v1.shape == v2.shape
    return all(torch.all((v1 == v2), dim=-1))


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("task", ["lm", "bidir-lm"])
def test_data_loader_char(shuffle, task):

    filepath = "data/test_data/6.csv"
    batch_sizes = (10, 10)
    tokenizer = CharTokenizer(None)
    data_handler, _ = create_data_loaders([filepath], batch_sizes, tokenizer, task, shuffle=shuffle)
    bidirectional = task == "bidir-lm"
    for batch in data_handler:
        x: torch.Tensor = batch["input"]
        x_length = batch["length"]
        for i in range(0, batch_sizes[0]):
            # Confirm that the targets are equal to the inputs shifted
            # by 1
            all(
                x[i, 1 : x_length[i] - int(bidirectional)] == batch["target"][i, : x_length[i] - 1 - int(bidirectional)]
            )


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("task", ["lm", "bidir-lm"])
@pytest.mark.parametrize("mode", ["field", "global"])
def test_data_loader_word(shuffle, task, mode):

    filepath = "data/test_data/6.csv"
    batch_sizes = (10, 10)
    vocab_file = f"data/vocab_{mode}_cutoff=40.json"

    if mode == "field":
        vocab = LANLVocab(vocab_file)
        tokenizer = LANLTokenizer(vocab)
    else:
        vocab = GlobalVocab(vocab_file)
        tokenizer = FieldTokenizer(vocab)

    data_handler, _ = create_data_loaders([filepath], batch_sizes, tokenizer, task, shuffle)
    bidirectional = task == "bidir-lm"
    expected_input_length = 10 - 1 if task == "lm" else 10 + 2
    for batch in data_handler:
        x: torch.Tensor = batch["input"]
        assert x.shape == torch.Size([batch_sizes[0], expected_input_length]), (
            "bidirectional" if bidirectional else "forward"
        )
        # Confirm that the targets are equal to the inputs shifted by 1
        assert batch_equal(
            batch["input"][:, 1 : batch["input"].shape[1] - int(bidirectional)],
            batch["target"][:, : batch["target"].shape[1] - int(not bidirectional)],
        ), f"{'bidir' if bidirectional else 'forward'}-shift"


def test_data_loader_tiered():
    pytest.skip()
