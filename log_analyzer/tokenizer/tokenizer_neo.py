import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
MSK_TOKEN = "[MSK]"
OOV_TOKEN = "[OOV]"
CLS_TOKEN = "[CLS]"

SPECIAL_TOKENS = [PAD_TOKEN]


class Tokenizer(ABC):

    pad_idx: int
    pad_token: str

    @abstractmethod
    def tokenize(self, line):
        ...

    @abstractmethod
    def encode(self, tokens):
        ...

    @abstractmethod
    def decode(self, indexes):
        ...


class CharTokenizer(Tokenizer):
    def __init__(self) -> None:
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MSK_TOKEN, CLS_TOKEN]

        self.special_tokens = {}
        for i, t in enumerate(special_tokens):
            self.special_tokens[t] = i

        self.offset = len(special_tokens)
        self.pad_token = PAD_TOKEN
        self.pad_idx = self.special_tokens[PAD_TOKEN]

    def encode(self, tokens):
        return [str(ord(c) + self.offset) for c in tokens]

    def decode(self, indexes):

        return [str(chr(i - self.offset)) for i in indexes]

    def tokenize(self, line):
        return self.encode(line)


class FieldVocab(ABC):

    vocab: dict
    special_tokens: dict
    size: int

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
        with open(vocab_file) as f:
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
        with open(vocab_file) as f:
            self.vocab = OrderedDict(json.load(f))

        # Vocab size including special tokens
        self.size = 0
        for mapping in self.vocab.items():
            self.size += len(mapping)

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

    def token2idx(self, token, field) -> int:
        """Returns the index of the given token for a given field."""
        try:
            return self.vocab[field][token]
        except KeyError:
            return self.oov_tokens[self.field_indexes[field]]

    def idx2token(self, idx) -> int:
        """Returns a token that maps to the given index."""

        def search(query, dictionary):
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

        with open(counts_file) as f:
            counts = json.load(f)

        # Add one Out-Of-Vocabulary and MASK index for each field
        vocab[MSK_TOKEN] = []
        vocab[OOV_TOKEN] = []
        for t in [OOV_TOKEN, MSK_TOKEN]:
            for i, _ in enumerate(counts):
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
        with open(outfile, "w") as f:
            json.dump(vocab, f, indent=" ")

        return cls(outfile)


class LANLTokenizer:
    def __init__(self, vocab: LANLVocab) -> None:
        self.vocab = vocab
        self.delimiter = ","
        self.num_fields = len(self.field_names)

    @property
    def num_special_tokens(self):
        return len(self.vocab.special_tokens)

    @property
    def vocab_size(self):
        return self.vocab.size

    @property
    def field_names(self):
        return self.vocab.field_names

    def tokenize(self, line):

        if isinstance(line, str):
            tokens = line.split(",")
        else:
            tokens = line

        return self.encode(tokens)

    def encode(self, tokens):

        if len(tokens) != len(self.field_names):
            raise RuntimeError("Number of fields in input does not match number of fields in vocabulary.")

        indexes = np.zeros(self.num_fields, dtype=np.int64)
        for i, field in enumerate(self.field_names):
            indexes[i] = self.vocab.token2idx(tokens[i], field)

        return indexes

    def decode(self, indexes: Iterable[int]):
        return [self.vocab.idx2token(i) for i in indexes]

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

        return masked_tokens, labels, sample_weights
