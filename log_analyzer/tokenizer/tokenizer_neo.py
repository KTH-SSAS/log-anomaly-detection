from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .vocab import CLS_TOKEN, EOS_TOKEN, MSK_TOKEN, PAD_TOKEN, SOS_TOKEN, GlobalVocab, LANLVocab


def mask_tokens(
    indexes: NDArray[np.int64],
    mask_idx,
    vocab_range: Tuple[int, int],
    percentage_to_mask=0.15,
    p_preserve=0.1,
    p_random=0.1,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Replace a percentage of the tokens with mask tokens.

    to_mask :       0110101010 <- 5 mask bits set to mask those tokens
    mask2mask:      0100101010 <- 1 mask bit cleared to preserve that token
    mask2random:    0000000010 <- 1 mask bit remains to set that token to a random token
    """

    masked_tokens = np.array(indexes, dtype=np.int64)

    # Unmasked tokens are used as labels
    labels = np.array(indexes, dtype=np.int64)
    length = len(indexes)

    # Positions in the sequence that will be masked
    to_mask = np.random.rand(length) < percentage_to_mask

    # Ensure that at least one token is masked
    if np.sum(to_mask) == 0:
        token_to_mask = np.random.randint(length)
        masked_tokens[token_to_mask] = mask_idx
        sample_weights = np.zeros(labels.shape, dtype=np.bool_)
        sample_weights[token_to_mask] = 1
        return masked_tokens, labels, sample_weights

    # Leave 10% of tokens unmasked
    inp_mask_2mask = to_mask & (np.random.rand(length) < 1 - p_preserve)

    # Use the mask token for the right field (if different mask tokens are used)
    masked_tokens[inp_mask_2mask] = mask_idx

    # Set 10% of tokens to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(length) < p_random)

    masked_tokens[inp_mask_2random] = np.random.randint(
        vocab_range[0],
        high=vocab_range[1],
        size=(inp_mask_2random.sum(),),
    )

    # Set targets to -1 by default
    labels_mask = -1 * np.ones(length, dtype=int)
    labels_mask[to_mask] = masked_tokens[to_mask]

    # Sample weights is a mask that shows which positions in the sequence were selected
    # to be masked.
    sample_weights = np.ones(labels_mask.shape, dtype=np.bool_)
    sample_weights[labels_mask == -1] = 0

    # Only consider masked positions in label
    labels *= sample_weights

    return masked_tokens, labels, sample_weights


class Tokenizer(ABC):

    jagged: bool
    pad_idx: int
    pad_token: str
    add_sos: bool
    add_eos: bool
    _num_users: int
    users: Dict[str, int]

    @abstractmethod
    def __init__(self, vocab, users: List[str] = None) -> None:
        super().__init__()

    @abstractmethod
    def tokenize(self, line: Union[str, List[str]], add_sos: bool = False, add_eos: bool = False) -> NDArray[np.int64]:
        ...

    @abstractmethod
    def encode(self, tokens: Union[str, List[str]], add_sos, add_eos) -> NDArray[np.int64]:
        ...

    @abstractmethod
    def decode(self, indexes: NDArray[np.int64]) -> List[str]:
        ...

    @abstractmethod
    def detokenize(self, indexes: NDArray[np.int64]) -> str:
        ...

    def user_idx(self, user):
        if self.num_users == 0:
            return 0
        return self.users[user]

    @abstractmethod
    def mask_tokens(
        self, indexes: NDArray[np.int64], percentage_to_mask=0.15, p_preserve=0.1, p_random=0.1
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        ...

    @property
    @abstractmethod
    def num_users(self) -> int:
        return self._num_users

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def sequence_length(self) -> Optional[int]:
        ...


class CharTokenizer(Tokenizer):
    def detokenize(self, indexes: NDArray[np.int64]) -> str:
        return "".join(self.decode(indexes))

    def __init__(self, vocab, users: List[str] = None) -> None:
        super().__init__(vocab, users)
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MSK_TOKEN, CLS_TOKEN]

        self.jagged = True

        self.special_tokens = {}
        for i, t in enumerate(special_tokens):
            self.special_tokens[t] = i

        self.offset = len(special_tokens)
        self.pad_token = PAD_TOKEN
        self.pad_idx = self.special_tokens[PAD_TOKEN]

        self.eos_idx = self.special_tokens[EOS_TOKEN]
        self.sos_idx = self.special_tokens[SOS_TOKEN]
        self.mask_idx = self.special_tokens[MSK_TOKEN]

        if users is not None:
            self.users = {users[i]: i for i in range(len(users))}
        else:
            self.users = {}

        self._num_users = len(self.users)

    @property
    def vocab_size(self):
        """There are 126 printable ASCII characters."""
        return 126 + self.offset

    @property
    def sequence_length(self):
        return None

    def encode(self, tokens, add_sos, add_eos):

        input_length = len(tokens)
        total_length = input_length + int(add_sos) + int(add_eos)

        indexes = np.zeros(total_length, dtype=np.int64)

        iterator = range(int(add_sos), input_length - int(add_eos))

        if add_sos:
            indexes[0] = self.sos_idx

        for i in iterator:
            indexes[i] = ord(tokens[i]) + self.offset

        if add_eos:
            indexes[-1] = self.eos_idx

        return indexes

    def decode(self, indexes):

        return [str(chr(i - self.offset)) for i in indexes]

    def tokenize(self, line=False, add_sos=False, add_eos=False):

        if isinstance(line, list):
            line = "".join(line)

        return self.encode(line, add_sos, add_eos)

    def mask_tokens(
        self, indexes: NDArray[np.int64], percentage_to_mask=0.15, p_preserve=0.1, p_random=0.1
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        return mask_tokens(
            indexes, self.mask_idx, (self.offset, self.vocab_size), percentage_to_mask, p_preserve, p_random
        )

    @property
    def num_users(self):
        return self._num_users


class LANLTokenizer(Tokenizer):
    """Tokenizer for LANL data."""

    def __init__(self, vocab, users: List[str] = None) -> None:
        super().__init__(vocab, users)
        self.vocab: LANLVocab = vocab
        self.delimiter = ","
        self.num_fields = len(self.field_names)
        self.jagged = False

        if users is not None:
            self.users = {users[i]: i for i in range(len(users))}
        else:
            self.users = {}

        self._num_users = len(self.users)

    @property
    def sequence_length(self):
        return self.num_fields

    @property
    def num_special_tokens(self):
        return len(self.vocab.special_tokens)

    @property
    def vocab_size(self):
        return self.vocab.size

    @property
    def field_names(self):
        return self.vocab.field_names

    @property
    def num_users(self):
        return self._num_users

    def tokenize(self, line, add_sos=False, add_eos=False):

        if isinstance(line, str):
            tokens = line.split(",")
        else:
            tokens = line

        return self.encode(tokens, add_sos, add_eos)

    def encode(self, tokens, add_sos, add_eos):

        if len(tokens) != len(self.field_names):
            raise RuntimeError("Number of fields in input does not match number of fields in vocabulary.")

        total_length = self.num_fields + int(add_sos) + int(add_eos)

        indexes = np.zeros(total_length, dtype=np.int64)

        iterator = range(int(add_sos), self.num_fields - int(add_eos))

        if add_sos:
            indexes[0] = self.vocab.sos_idx

        for i in iterator:
            field = self.field_names[i]
            indexes[i] = self.vocab.token2idx(tokens[i], field)

        if add_eos:
            indexes[-1] = self.vocab.eos_idx

        return indexes

    def decode(self, indexes: Iterable[int]) -> List[str]:
        return [self.vocab.idx2token(i) for i in indexes]

    def detokenize(self, indexes) -> str:
        return ",".join(self.decode(indexes))

    def mask_tokens(
        self, indexes: NDArray[np.int64], percentage_to_mask=0.2, p_preserve=0.1, p_random=0.1
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Replace a percentage of the tokens with mask tokens.

        to_mask :       0110101010 <- 5 mask bits set to mask those tokens
        mask2mask:      0100101010 <- 1 mask bit cleared to preserve that token
        mask2random:    0000000010 <- 1 mask bit remains to set that token to a random token
        """

        masked_tokens = np.array(indexes, dtype=np.int64)

        # Unmasked tokens are used as labels
        labels = np.array(indexes, dtype=np.int64)
        length = len(indexes)

        # Positions in the sequence that will be masked
        to_mask = np.random.rand(length) < percentage_to_mask

        # Ensure that at least one token is masked
        if np.sum(to_mask) == 0:
            token_to_mask = np.random.randint(length)
            masked_tokens[token_to_mask] = self.vocab.mask_tokens[token_to_mask]
            sample_weights = np.zeros(labels.shape, dtype=np.bool_)
            sample_weights[token_to_mask] = 1
            return masked_tokens, labels, sample_weights

        # Leave 10% of tokens unmasked
        inp_mask_2mask = to_mask & (np.random.rand(length) < 1 - p_preserve)

        # Use the mask token for the right field (if different mask tokens are used)
        masked_tokens[inp_mask_2mask] = self.vocab.mask_tokens[inp_mask_2mask]

        # Set 10% of tokens to a random token
        inp_mask_2random = inp_mask_2mask & (np.random.rand(length) < p_random)

        if np.any(inp_mask_2random):
            low = self.vocab.field_vocab_min[inp_mask_2random]
            high = self.vocab.field_vocab_max[inp_mask_2random]

            # This is only true when fields have a vocab size of 1
            if np.all(low == high):
                masked_tokens[inp_mask_2random] = self.vocab.field_vocab_min[inp_mask_2random]
            elif np.any(low == high):
                raise NotImplementedError(
                    "The case where one field has a single token in its vocabulary is unsupported."
                )
            else:
                masked_tokens[inp_mask_2random] = np.random.randint(
                    low=low,
                    high=high + 1,  # +1 since randint is exclusive of high value
                )

        # Set targets to -1 by default
        labels_mask = -1 * np.ones(length, dtype=int)
        labels_mask[to_mask] = masked_tokens[to_mask]

        # Sample weights is a mask that shows which positions in the sequence were selected
        # to be masked.
        sample_weights = np.ones(labels_mask.shape, dtype=np.bool_)
        sample_weights[labels_mask == -1] = 0

        # Only consider masked positions in label
        labels *= sample_weights

        return masked_tokens, labels, sample_weights


class FieldTokenizer(Tokenizer):
    """Tokenizes the fields of a log line."""

    def detokenize(self, indexes) -> str:
        return ",".join(self.decode(indexes))

    def __init__(self, vocab: GlobalVocab, users=None) -> None:
        super().__init__(vocab, users)
        self.vocab = vocab
        self.mask_idx = vocab.mask_idx
        self.offset = vocab.num_special_tokens
        self.jagged = False

        if users is not None:
            self.users = {users[i]: i for i in range(len(users))}
        else:
            self.users = {}

        self._num_users = len(self.users)

    def encode(self, tokens, add_sos, add_eos):
        input_length = len(tokens)
        total_length = input_length + int(add_sos) + int(add_eos)

        indexes = np.zeros(total_length, dtype=np.int64)

        iterator = range(int(add_sos), input_length - int(add_eos))

        if add_sos:
            indexes[0] = self.vocab.sos_idx

        for i in iterator:
            indexes[i] = self.vocab.token2idx(tokens[i], "")

        if add_eos:
            indexes[-1] = self.vocab.eos_idx

        return indexes

    def decode(self, indexes: Iterable[int]) -> List[str]:
        return [self.vocab.idx2token(i) for i in indexes]

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def sequence_length(self):
        return None

    def tokenize(self, line, add_sos=False, add_eos=False):
        if isinstance(line, str):
            tokens = line.split(",")
        else:
            tokens = line
        return self.encode(tokens, add_sos, add_eos)

    @property
    def vocab_size(self):
        return self.vocab.size

    def mask_tokens(
        self, indexes: NDArray[np.int64], percentage_to_mask=0.15, p_preserve=0.1, p_random=0.1
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        return mask_tokens(
            indexes, self.mask_idx, (self.offset, self.vocab_size), percentage_to_mask, p_preserve, p_random
        )
