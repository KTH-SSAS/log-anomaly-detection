import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List

import numpy as np

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
MSK_TOKEN = "[MSK]"
OOV_TOKEN = "[OOV]"
CLS_TOKEN = "[CLS]"

SPECIAL_TOKENS = [PAD_TOKEN]


class Tokenizer(ABC):

    jagged: bool
    pad_idx: int
    pad_token: str
    add_sos: bool
    add_eos: bool
    _num_users: int
    users: dict[str, int]

    @abstractmethod
    def __init__(self, vocab, users: List[str] = None) -> None:
        super().__init__()

    @abstractmethod
    def tokenize(self, line, add_sos=False, add_eos=False):
        ...

    @abstractmethod
    def encode(self, tokens, add_sos, add_eos):
        ...

    @abstractmethod
    def decode(self, indexes):
        ...

    @abstractmethod
    def mask_tokens(self, tokens: list, percentage_to_mask=0.15, p_preserve=0.1, p_random=0.1):
        ...

    def user_idx(self, user):
        if self.num_users == 0:
            return 0
        return self.users[user]

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
    def sequence_length(self):
        ...


class CharTokenizer(Tokenizer):
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

    def mask_tokens(self, tokens: list, percentage_to_mask=0.15, p_preserve=0.1, p_random=0.1):
        """Replace a percentage of the tokens with mask tokens.

        to_mask :       0110101010 <- 5 mask bits set to mask those tokens
        mask2mask:      0100101010 <- 1 mask bit cleared to preserve that token
        mask2random:    0000000010 <- 1 mask bit remains to set that token to a random token
        """

        masked_tokens = np.array(tokens, dtype=np.int64)

        # Unmasked tokens are used as labels
        labels = np.array(tokens, dtype=np.int64)
        length = len(tokens)

        # Positions in the sequence that will be masked
        to_mask = np.random.rand(length) < percentage_to_mask

        # Ensure that at least one token is masked
        if np.sum(to_mask) == 0:
            token_to_mask = np.random.randint(length)
            masked_tokens[token_to_mask] = self.mask_idx
            sample_weights = np.zeros(labels.shape, dtype=np.bool_)
            sample_weights[token_to_mask] = 1
            return masked_tokens, labels, sample_weights

        # Leave 10% of tokens unmasked
        inp_mask_2mask = to_mask & (np.random.rand(length) < 1 - p_preserve)

        # Use the mask token for the right field (if different mask tokens are used)
        masked_tokens[inp_mask_2mask] = self.mask_idx

        # Set 10% of tokens to a random token
        inp_mask_2random = inp_mask_2mask & (np.random.rand(length) < p_random)

        masked_tokens[inp_mask_2random] = np.random.randint(
            self.offset,
            high=self.vocab_size,
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

    @property
    def num_users(self):
        return self._num_users


class FieldVocab(ABC):

    special_tokens: dict
    size: int
    num_users: int
    vocab: dict

    @abstractmethod
    def __init__(self, vocab_file: str) -> None:
        ...

    @abstractmethod
    def token2idx(self, token: str, field: str):
        ...

    @abstractmethod
    def idx2token(self, idx: int):
        ...

    @classmethod
    @abstractmethod
    def counts2vocab(cls, counts_file: str, outfile: str, cutoff: int):
        ...


class GlobalVocab(FieldVocab):
    def __init__(self, vocab_file) -> None:
        super().__init__(vocab_file)
        with open(vocab_file, encoding="utf8") as f:
            self.vocab = json.load(f)

    def token2idx(self, token, _):
        return self.vocab[token]

    def idx2token(self, idx):
        for k, v in self.vocab.items():
            if idx == v:
                return k
        return OOV_TOKEN

    @classmethod
    def counts2vocab(cls, counts_file, outfile, cutoff):
        raise NotImplementedError("Not implemented.")


class LANLVocab(FieldVocab):
    def __init__(self, vocab_file) -> None:
        super().__init__(vocab_file)
        with open(vocab_file, encoding="utf8") as f:
            self.vocab = OrderedDict(json.load(f))

        # Vocab size including special tokens
        self.size = 0
        for token_set in self.vocab.values():
            self.size += len(token_set)

        self.special_tokens = self.vocab["special_tokens"]
        del self.vocab["special_tokens"]

        self.mask_tokens = np.array(self.vocab[MSK_TOKEN])
        self.oov_tokens = np.array(self.vocab[OOV_TOKEN])

        del self.vocab[OOV_TOKEN]
        del self.vocab[MSK_TOKEN]

        # Save field names, without special keys
        self.field_names = list(self.vocab.keys())
        self.field_indexes = {field: i for i, field in enumerate(self.vocab)}

        # Precalculated vocab limits to help generating random tokens for each field
        self.field_vocab_max = np.zeros(len(self.field_names))
        for i, field in enumerate(self.field_names):
            self.field_vocab_max[i] = max(self.vocab[field].values())

        self.field_vocab_min = np.zeros(len(self.field_names))
        for i, field in enumerate(self.field_names):
            self.field_vocab_min[i] = min(self.vocab[field].values())

        self._eos_idx = self.special_tokens[EOS_TOKEN]
        self._sos_idx = self.special_tokens[SOS_TOKEN]

    @property
    def num_users(self):
        """Count the OOV token as a user, but not the mask token."""
        return len(self.vocab["src_user"]) - 1

    @property
    def eos_idx(self):
        return self._eos_idx

    @property
    def sos_idx(self):
        return self._sos_idx

    def token2idx(self, token, field) -> int:
        """Returns the index of the given token for a given field."""
        try:
            return self.vocab[field][token]
        except KeyError:
            return self.oov_tokens[self.field_indexes[field]]

    def idx2token(self, idx) -> str:
        """Returns a token that maps to the given index."""

        def search(query, dictionary: dict):
            for k, i in dictionary.items():
                if i == query:
                    return True, k
            return False, None

        # Search through the field dictionaries with different threads
        threads = []
        with ThreadPoolExecutor() as executor:
            for field in self.vocab:
                threads.append(executor.submit(search, idx, self.vocab[field]))

        for t in threads:
            found, token = t.result()
            if found:
                return token

        raise KeyError("Index not present in vocabulary.")

    @classmethod
    def counts2vocab(cls, counts_file, outfile, cutoff):
        """Generates a vocabulary file based on a file of token counts per
        field.

        Tokens are assigned different indexes for each field they appear
        in.
        """

        # Use an ordered dict to maintain the order of field names
        vocab = OrderedDict()

        # Add special tokens to entry unrelated to fields
        vocab["special_tokens"] = {}
        index = 0
        for t in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, CLS_TOKEN]:
            vocab["special_tokens"][t] = index
            index += 1

        with open(counts_file, encoding="utf8") as f:
            counts = json.load(f)

        # Add one Out-Of-Vocabulary and MASK index for each field
        vocab[MSK_TOKEN] = []
        vocab[OOV_TOKEN] = []
        for t in [OOV_TOKEN, MSK_TOKEN]:
            for _ in counts:
                vocab[t].append(index)
                index += 1

        for field in counts:
            vocab[field] = {}

            # Add indexes for the tokens in the field
            for token, count in counts[field].items():
                if count > cutoff:
                    vocab[field][token] = index
                    index += 1

        vocab.move_to_end(MSK_TOKEN)
        vocab.move_to_end(OOV_TOKEN)
        vocab.move_to_end("special_tokens")

        print(f"Generated vocab with {index} words.")

        with open(outfile, "w", encoding="utf8") as f:
            json.dump(vocab, f, indent=" ")

        return cls(outfile)


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

    def mask_tokens(self, tokens: list, percentage_to_mask=0.2, p_preserve=0.1, p_random=0.1):
        """Replace a percentage of the tokens with mask tokens.

        to_mask :       0110101010 <- 5 mask bits set to mask those tokens
        mask2mask:      0100101010 <- 1 mask bit cleared to preserve that token
        mask2random:    0000000010 <- 1 mask bit remains to set that token to a random token
        """

        masked_tokens = np.array(tokens, dtype=np.int64)

        # Unmasked tokens are used as labels
        labels = np.array(tokens, dtype=np.int64)
        length = len(tokens)

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
