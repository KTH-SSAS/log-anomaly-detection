import pytest
import torch

from log_analyzer.config.trainer_config import DataConfig
from log_analyzer.data.data_loader import TieredTransformerBatcher

def batch_equal(v1: torch.Tensor, v2: torch.Tensor):
    assert v1.shape == v2.shape
    return all(torch.all((v1 == v2), dim=-1))

@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_tiered_data_loader_word(shuffle, bidirectional):
    from log_analyzer.train_loop import calculate_max_input_length

    filepath = "data/test_data/word_day_split/0.txt"
    data_config = DataConfig.init_from_file("config/lanl_config_data_word.json")
    batch_size = 10
    skip_sos = False
    jagged = False
    model_dim = 128
    context_model_dim = 128
    context_input_dimension = 128
    shift_window = 100
    num_steps = 3
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler = TieredTransformerBatcher(
            filepath,
            data_config.sentence_length,
            model_dim,
            context_model_dim,
            skip_sos,
            jagged,
            bidirectional,
            context_input_dimension,
            shift_window=shift_window,
            batch_size=batch_size,
            num_steps=num_steps,
            delimiter=" ",)

    for batch in data_handler:
        x: torch.Tensor = batch["input"]
        assert x.shape == torch.Size([batch_size, input_length]), "bidirectional" if bidirectional else "forward"
        assert batch_equal(
            batch["input"][:, 1 : batch["input"].shape[1] - int(bidirectional)],
            batch["target"][:, : batch["target"].shape[1] - int(not bidirectional)],
        ), f"{'bidir' if bidirectional else 'forward'}-shift"  # Confirm that the targets are equal to the inputs shifted by 1
