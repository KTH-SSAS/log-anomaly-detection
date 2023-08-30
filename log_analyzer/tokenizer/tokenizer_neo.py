from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .vocab import CLS_TOKEN, EOS_TOKEN, MSK_TOKEN, PAD_TOKEN, SOS_TOKEN, FieldVocab, GlobalVocab, LANLVocab

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600

def mask_tokens(
    indexes: NDArray[np.int64],
    mask_idx: int,
    test: bool = False,
) -> Tuple[NDArray, NDArray]:
    """Replace a random token with a mask token. If test=True instead each token will be replaced with a mask token,
    one at a time, generatic (seq_len) new sequences each with 1 mask token.

    If test=True: if 0 is the mask token, and original sequence is 1,2,3,4,5, then the output will be:
    0,2,3,4,5
    1,0,3,4,5
    1,2,0,4,5
    1,2,3,0,5
    1,2,3,4,0
    """
    masked_tokens = np.array(indexes, dtype=np.int64)
    length = len(indexes)

    if not test:
        # Randomly mask a single token
        mask_pos = np.random.randint(0, length)
        masked_tokens[mask_pos] = mask_idx
        labels = np.zeros_like(indexes)
        labels[mask_pos] = indexes[mask_pos]
    else:
        # Mask each token in turn
        masked_tokens = np.repeat(np.expand_dims(masked_tokens, axis=0), length, axis=0)
        # Positions in the sequence that will be masked
        for i in range(length):
            masked_tokens[i, i] = mask_idx
        # Unmasked tokens are used as labels
        labels = np.array(indexes, dtype=np.int64)

    return masked_tokens, labels


class Tokenizer(ABC):

    jagged: bool
    pad_idx: int
    pad_token: str
    add_sos: bool
    add_eos: bool
    _num_users: int
    users: Dict[str, int]
    vocab: FieldVocab

    @abstractmethod
    def __init__(self, vocab, users: List[str] = None, include_timestamp: bool = False) -> None:
        super().__init__()

    @abstractmethod
    def tokenize(self, line: Union[str, List[str]], add_sos: bool = False, add_eos: bool = False) -> NDArray[np.int64]:
        ...

    @abstractmethod
    def _encode_single_position(self, token: str, field: int):
        ...

    def _encode_timestamp(self, token: str, field: int):
        # Bucket the timestamp into 1 hour intervals
        hour_bucket = (int(token) % SECONDS_PER_DAY) // (1 * SECONDS_PER_HOUR)
        return self.vocab.token2idx("T" + str(hour_bucket), field)

    def encode(self, tokens: Union[str, List[str]], add_sos, add_eos) -> NDArray[np.int64]:

        num_pre_tokens = int(add_sos)
        num_post_tokens = int(add_eos)

        total_length = num_pre_tokens + len(tokens) + num_post_tokens

        indexes = np.zeros(total_length, dtype=np.int64)

        for i in range(num_pre_tokens):
            indexes[i] = self.sos_idx

        for i in range(num_pre_tokens, num_pre_tokens + self.include_timestamp):
            indexes[i] = self._encode_timestamp(tokens[i - num_pre_tokens], i - num_pre_tokens)

        for i in range(num_pre_tokens + self.include_timestamp, total_length - num_post_tokens):
            # If we're not including timestamp, we need to shift the field index by 1
            indexes[i] = self._encode_single_position(tokens[i - num_pre_tokens], i - num_pre_tokens + (1 - self.include_timestamp))

        for i in range(num_post_tokens):
            indexes[total_length - num_post_tokens + i] = self.eos_idx

        return indexes

    @abstractmethod
    def decode(self, indexes: NDArray[np.int64]) -> List[str]:
        ...

    @abstractmethod
    def detokenize(self, indexes: NDArray[np.int64]) -> str:
        ...

    @property
    @abstractmethod
    def sos_idx(self):
        ...

    @property
    @abstractmethod
    def eos_idx(self):
        ...

    @property
    @abstractmethod
    def include_timestamp(self):
        ...

    def user_idx(self, user):
        if self.num_users == 0:
            # Can't return user_idx if we don't have a list of users
            raise RuntimeError(
                "Cannot return user index since no counts file was provided during tokenizer initialisation"
            )
        return self.users[user]

    @abstractmethod
    def mask_tokens(
        self, indexes: NDArray[np.int64], test: bool) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
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

    def __init__(self, vocab, users: List[str] = None, include_timestamp: bool = False) -> None:
        super().__init__(vocab, users)
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MSK_TOKEN, CLS_TOKEN]

        if include_timestamp:
            raise UserWarning("Timestamps are not supported for character tokenization")
        self._include_timestamp = False

        self.jagged = True

        self.special_tokens = {}
        for i, t in enumerate(special_tokens):
            self.special_tokens[t] = i

        self.offset = len(special_tokens)
        self.pad_token = PAD_TOKEN
        self.pad_idx = self.special_tokens[PAD_TOKEN]

        self._eos_idx = self.special_tokens[EOS_TOKEN]
        self._sos_idx = self.special_tokens[SOS_TOKEN]
        self.mask_idx = self.special_tokens[MSK_TOKEN]

        if users is not None:
            self.users = {users[i]: i for i in range(len(users))}
        else:
            self.users = {}

        self._num_users = len(self.users)

    @property
    def eos_idx(self):
        return self._eos_idx

    @property
    def sos_idx(self):
        return self._sos_idx

    @property
    def include_timestamp(self):
        return self._include_timestamp

    @property
    def vocab_size(self):
        """There are 126 printable ASCII characters."""
        return 126 + self.offset

    @property
    def sequence_length(self):
        return None

    def _encode_single_position(self, token, field):
        return ord(token) + self.offset

    def decode(self, indexes):

        return [str(chr(i - self.offset)) for i in indexes]

    def tokenize(self, line=False, add_sos=False, add_eos=False):

        if isinstance(line, list):
            line = "".join(line)

        return self.encode(line, add_sos, add_eos)

    def mask_tokens(
        self, indexes: NDArray[np.int64], test: bool = False
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        return mask_tokens(
            indexes, self.mask_idx, test
        )

    @property
    def num_users(self):
        return self._num_users


class LANLTokenizer(Tokenizer):
    """Tokenizer for LANL data."""

    def __init__(self, vocab, users: List[str] = None, include_timestamp: bool = False) -> None:
        super().__init__(vocab, users)
        self.vocab: LANLVocab = vocab
        self.delimiter = ","
        self.num_fields = 10  # From LANL log data
        self.jagged = False
        self._include_timestamp = include_timestamp

        if users is not None:
            self.users = {users[i]: i for i in range(len(users))}
        else:
            self.users = {}

        self._num_users = len(self.users)

    @property
    def sos_idx(self):
        return self.vocab.sos_idx

    @property
    def eos_idx(self):
        return self.vocab.eos_idx

    @property
    def include_timestamp(self):
        return self._include_timestamp

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
        """Expects a line in the format 'src_user,src_domain,dst_user,dst_domai
        n,src_pc,dst_pc,auth_type,logon_type,auth_orient,success'."""
        if isinstance(line, str):
            tokens = line.split(",")
        else:
            tokens = line

        return self.encode(tokens, add_sos, add_eos)

    def _encode_single_position(self, token, field):
        return self.vocab.token2idx(token, field)

    def decode(self, indexes: Iterable[int]) -> List[str]:
        return [self.vocab.idx2token(idx, field + (1 - self.include_timestamp)) for field, idx in enumerate(indexes)]

    def detokenize(self, indexes) -> str:
        return ",".join(self.decode(indexes))

    def mask_tokens(
        self, indexes: NDArray[np.int64], test: bool = False
    ) -> Tuple[NDArray, NDArray]:
        """Replace a random token with a mask token. If test=True instead each token will be replaced with a mask token,
        one at a time, generatic (seq_len) new sequences each with 1 mask token.

        If test=True: if 0 is the mask token, and original sequence is 1,2,3,4,5, then the output will be:
        0,2,3,4,5
        1,0,3,4,5
        1,2,0,4,5
        1,2,3,0,5
        1,2,3,4,0
        """
        masked_tokens = np.array(indexes, dtype=np.int64)
        length = len(indexes)

        if not test:
            # Randomly mask a single token
            mask_pos = np.random.randint(0, length)
            masked_tokens[mask_pos] = self.vocab.mask_tokens[mask_pos]
            labels = np.zeros_like(indexes)
            labels[mask_pos] = indexes[mask_pos]
        else:
            # Mask each token in turn
            masked_tokens = np.repeat(np.expand_dims(masked_tokens, axis=0), length, axis=0)
            # Positions in the sequence that will be masked
            for i in range(length):
                masked_tokens[i, i] = self.vocab.mask_tokens[i]
            # Unmasked tokens are used as labels
            labels = np.array(indexes, dtype=np.int64)

        return masked_tokens, labels


class GlobalTokenizer(Tokenizer):
    """Tokenizes the fields of a log line."""

    def detokenize(self, indexes) -> str:
        return ",".join(self.decode(indexes))

    @property
    def sos_idx(self):
        return self.vocab.sos_idx

    @property
    def eos_idx(self):
        return self.vocab.eos_idx

    @property
    def include_timestamp(self):
        return self._include_timestamp

    def __init__(self, vocab: GlobalVocab, users=None, include_timestamp: bool = False) -> None:
        super().__init__(vocab, users)
        self.vocab: GlobalVocab = vocab
        self.mask_idx = vocab.mask_idx
        self.offset = vocab.num_special_tokens
        self.jagged = False
        self._include_timestamp = include_timestamp

        if users is not None:
            self.users = {users[i]: i for i in range(len(users))}
        else:
            self.users = {}

        self._num_users = len(self.users)

    def _encode_single_position(self, token, field):
        return self.vocab.token2idx(token, field)

    def decode(self, indexes: Iterable[int]) -> List[str]:
        return [self.vocab.idx2token(idx, field + (1 - self.include_timestamp)) for field, idx in enumerate(indexes)]

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

    def mask_tokens(self, indexes: NDArray[np.int64], test: bool = False) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        return mask_tokens(indexes, self.mask_idx, test=test)
