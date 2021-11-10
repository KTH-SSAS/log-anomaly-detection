import torch
from torch.utils.data import DataLoader

from log_analyzer.config.trainer_config import DataConfig
from log_analyzer.data.data_loader import IterableLogDataset, MapLogDataset, create_data_loader

import pytest


def batch_equal(v1: torch.Tensor, v2: torch.Tensor):
    assert v1.shape == v2.shape
    return all(torch.all((v1 == v2), dim=-1))


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_data_loader_char(shuffle, bidirectional):
    from log_analyzer.train_loop import calculate_max_input_length

    filepath = "data/test_data/char_day_split/0.txt"
    data_config = DataConfig.init_from_file("config/lanl_char_config_data.json")
    batch_size = 10
    skip_sos = False
    jagged = True
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler = create_data_loader(filepath, batch_size, bidirectional, skip_sos, jagged, input_length, shuffle)
    for batch in data_handler:
        x: torch.Tensor = batch["input"]
        x_length = batch["length"]
        for i in range(0, batch_size):
            # Confirm that the targets are equal to the inputs shifted
            # by 1
            all(
                batch["input"][i, 1 : x_length[i] - int(bidirectional)]
                == batch["target"][i, : x_length[i] - 1 - int(bidirectional)]
            )


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_data_loader_word(shuffle, bidirectional):
    from log_analyzer.train_loop import calculate_max_input_length

    filepath = "data/test_data/word_day_split/0.txt"
    data_config = DataConfig.init_from_file("config/lanl_word_config_data.json")
    batch_size = 10
    skip_sos = False
    jagged = False
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler = create_data_loader(
        filepath, batch_size, bidirectional, skip_sos, jagged, data_config.sentence_length, shuffle
    )
    for batch in data_handler:
        x: torch.Tensor = batch["input"]
        assert x.shape == torch.Size([batch_size, input_length]), "bidirectional" if bidirectional else "forward"
        assert batch_equal(
            batch["input"][:, 1 : batch["input"].shape[1] - int(bidirectional)],
            batch["target"][:, : batch["target"].shape[1] - int(not bidirectional)],
        ), f"{'bidir' if bidirectional else 'forward'}-shift"  # Confirm that the targets are equal to the inputs shifted by 1


def test_data_loader_tiered():
    pytest.skip()
