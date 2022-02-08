import pytest
import torch

from log_analyzer.config.trainer_config import DataConfig
from log_analyzer.data.data_loader import TieredTransformerBatcher

@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_tiered_data_loader_word(shuffle, bidirectional):

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
