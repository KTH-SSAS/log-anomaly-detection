from log_analyzer.config.trainer_config import DataConfig
from torch.utils.data import DataLoader
from log_analyzer.data.data_loader import IterableLogDataSet, create_data_loader
import torch

def batch_equal(v1: torch.Tensor, v2: torch.Tensor):
    assert v1.shape == v2.shape
    return all(torch.all((v1 == v2), dim=-1))

def test_data_loader_char():
    from log_analyzer.train_loop import calculate_max_input_length
    filepath = 'data/test_data/char_day_split/0.txt'
    data_config = DataConfig.init_from_file('config/lanl_char_config_data.json')
    batch_size = 10
    bidirectional = False
    skip_sos = False
    jagged = True
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler = create_data_loader(filepath, batch_size, bidirectional, skip_sos, jagged, input_length)
    for batch in data_handler:
        x: torch.Tensor = batch['input']
        x_length = batch['length']
        for i in range(0, batch_size):           
            all(batch['input'][i, 1:x_length[i]] == batch['target'][i, :x_length[i]-1]) # Confirm that the targets are equal to the inputs shifted by 1

    # Test bidirectional
    bidirectional = True

    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler = create_data_loader(filepath, batch_size, bidirectional, skip_sos, jagged, input_length)
    for batch in data_handler:
        x: torch.Tensor = batch['input']
        x_length = batch['length']
        for i in range(0, batch_size):           
            all(batch['input'][i, 1:x_length[i]-1] == batch['target'][i, :x_length[i]-2]) # Confirm that the targets are equal to the inputs shifted by 1


        

def test_data_loader_word():
    from log_analyzer.train_loop import calculate_max_input_length
    filepath = 'data/test_data/word_day_split/0.txt'
    data_config = DataConfig.init_from_file('config/lanl_word_config_data.json')
    batch_size = 10
    bidirectional = False
    skip_sos = False
    jagged = False
    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler = create_data_loader(filepath, batch_size, bidirectional, skip_sos, jagged, data_config.sentence_length)
    for batch in data_handler:
        x: torch.Tensor = batch['input']
        assert x.shape == torch.Size([batch_size, input_length]), "forward"
        assert batch_equal(batch['input'][:, 1:], batch['target'][:, :-1]), "forward-shift" # Confirm that the targets are equal to the inputs shifted by 1

    # Test bidirectional
    bidirectional = True
    skip_sos = False

    input_length = calculate_max_input_length(data_config.sentence_length, bidirectional, skip_sos)
    data_handler = create_data_loader(filepath, batch_size, bidirectional, skip_sos, jagged, data_config.sentence_length)
    for batch in data_handler:
        x: torch.Tensor = batch['input']
        assert x.shape == torch.Size([batch_size, input_length]), "bidirectional"
        assert batch_equal(batch['input'][:, 1:-1], batch['target']), "bidir-shift" # The target should be equal to the input without the first and last token.


def test_data_loader_tiered():
    pass
