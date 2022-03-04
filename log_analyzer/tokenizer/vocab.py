import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
MSK_TOKEN = "[MSK]"
OOV_TOKEN = "[OOV]"
CLS_TOKEN = "[CLS]"

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
    """Vocabulary that keeps a single record of each field entry, irrespective
    of it's place in the log line."""

    special_tokens = {
        PAD_TOKEN: 0,
        SOS_TOKEN: 1,
        EOS_TOKEN: 2,
        CLS_TOKEN: 3,
        MSK_TOKEN: 4,
        OOV_TOKEN: 5,
    }

    def __init__(self, vocab_file) -> None:
        super().__init__(vocab_file)
        with open(vocab_file, encoding="utf8") as f:
            self.vocab = json.load(f)

        self.mask_idx = GlobalVocab.special_tokens[MSK_TOKEN]
        self.eos_idx = GlobalVocab.special_tokens[EOS_TOKEN]
        self.sos_idx = GlobalVocab.special_tokens[SOS_TOKEN]

        self.size = len(self.vocab)

        self.num_special_tokens = len(GlobalVocab.special_tokens)

    def token2idx(self, token, field) -> int:
        try:
            return self.vocab[token]
        except KeyError:
            return self.special_tokens[OOV_TOKEN]

    def idx2token(self, idx):
        for k, v in self.vocab.items():
            if idx == v:
                return k
        return OOV_TOKEN

    @classmethod
    def counts2vocab(cls, counts_file, outfile, cutoff):

        vocab = {}
        index = 0

        # Add special tokens to entry unrelated to fields
        for k, v in cls.special_tokens.items():
            vocab[k] = v
            index += 1

        with open(counts_file, encoding="utf8") as f:
            counts: dict[str, dict] = json.load(f)

        for field in counts:
            # Add indexes for the tokens in the field
            for token, count in counts[field].items():
                if count > cutoff and token not in vocab:
                    vocab[token] = index
                    index += 1

        print(f"Generated vocab with {index} words.")

        with open(outfile, "w", encoding="utf8") as f:
            json.dump(vocab, f, indent=" ")


class LANLVocab(FieldVocab):
    """Vocabulary that maintains a seperate wordlist for each log field."""

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
