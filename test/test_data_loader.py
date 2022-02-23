import pytest
import torch

from log_analyzer.config.trainer_config import DataConfig
from log_analyzer.data.data_loader import create_data_loaders, create_data_loaders_linelevel


def batch_equal(v1: torch.Tensor, v2: torch.Tensor):
    assert v1.shape == v2.shape
    return all(torch.all((v1 == v2), dim=-1))


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_data_loader_char(shuffle, bidirectional):
    from log_analyzer.train_loop import calculate_max_input_length

    filepath = "data/test_data/char_day_split/0.txt"
    data_config = DataConfig.init_from_file("config/lanl_config_data_char.json")
    batch_size = 10
    skip_sos = False
    jagged = True
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler, _ = create_data_loaders(
        filepath, batch_size, bidirectional, skip_sos, jagged, input_length, shuffle=shuffle
    )
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
    data_config = DataConfig.init_from_file("config/lanl_config_data_word.json")
    batch_size = 10
    skip_sos = False
    jagged = False
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler, _ = create_data_loaders(
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


def test_data_loader_loglinelevel():
    from log_analyzer.train_loop import calculate_max_input_length

    filepath = "data/test_data/word_day_split/0.txt"
    data_config = DataConfig.init_from_file("config/lanl_config_data_word.json")
    batch_size = 3
    window_size = 5
    skip_sos = True
    jagged = False
    bidirectional = False
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler, _ = create_data_loaders_linelevel(
        filepath, batch_size, bidirectional, skip_sos, jagged, window_size
    )
    assert len(data_handler) >= batch_size, "Dataset too small"
    for idx, batch in enumerate(data_handler):
        if idx == len(data_handler) - 1:
            assert batch["input"].shape[1:] == torch.Size([window_size, input_length])
        else:
            assert batch["input"].shape == torch.Size([batch_size, window_size, input_length])
        for b in range (batch["input"].shape[0]):
            assert batch_equal(
                batch["input"][b,1:],
                batch["target"][b,:-1],
            ), "forward-shift"  # Confirm that the targets are equal to the inputs shifted by 1