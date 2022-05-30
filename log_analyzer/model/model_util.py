import logging
import signal

import numpy as np
import torch
import torch.nn.init
from torch import nn


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def xavier_normal_(activation_function="relu"):
    return lambda x: torch.nn.init.xavier_normal_(x, gain=torch.nn.init.calculate_gain(activation_function))


def kaiming_normal_():
    return lambda x: torch.nn.init.kaiming_normal_(x, nonlinearity="relu")


def initialize_weights(net, initrange=1.0, dist_func=truncated_normal_):
    """Initializes the weights of the network using the given distribution
    Distribtuion can be either 'truncated', 'xavier', or 'kaiming."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            this_initrange = initrange * 1.0 / np.sqrt(m.weight.data.shape[1])
            m.weight.data = this_initrange * dist_func(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            dist_func(m.weight.data)


class DelayedKeyboardInterrupt:
    """Context handler for creating un-interruptable code blocks.

    Source: Stack overflow ("how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py")
    """

    def __init__(self):
        self.signal_received = False
        self.old_handler = None

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt.")

    def __exit__(self, *_):
        # Ignore input arguments (type, value, traceback)
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
