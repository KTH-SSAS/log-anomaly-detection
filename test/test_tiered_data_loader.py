import pytest
import torch

from log_analyzer.config.trainer_config import DataConfig
from log_analyzer.data.data_loader import TieredTransformerBatcher

@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
