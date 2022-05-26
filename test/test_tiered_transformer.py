import torch
from torch.nn.utils.rnn import pad_sequence

from log_analyzer.config.model_config import TieredTransformerConfig
from log_analyzer.model.transformer import TieredTransformer


def test_tiered_transformer_forward_word(
    test_tiered_transformer_config: TieredTransformerConfig, test_tiered_transformer_input, context_history
):
    consecutive_log = test_tiered_transformer_input[1].shape[0]
    batch_size = test_tiered_transformer_input[1].shape[1]
    sequence_length = test_tiered_transformer_input[1].shape[2]
    shift_window = test_tiered_transformer_config.shift_window
    vocab_size = test_tiered_transformer_config.vocab_size

    tieredTransformer = TieredTransformer(test_tiered_transformer_config, bidirectional=False)
    ctx_lengths_before_run = pad_sequence(
        tieredTransformer.get_ctx_data(test_tiered_transformer_input[0], test_tiered_transformer_input[1].device),
        batch_first=True,
    ).shape[1]
    token_output, _ = tieredTransformer(test_tiered_transformer_input, context_history)
    ctx_lengths_after_run = pad_sequence(
        tieredTransformer.get_ctx_data(test_tiered_transformer_input[0], test_tiered_transformer_input[1].device),
        batch_first=True,
    ).shape[1]
    assert min(ctx_lengths_before_run + consecutive_log, shift_window) == ctx_lengths_after_run
    assert token_output.shape == torch.Size([consecutive_log, batch_size, sequence_length, vocab_size])
