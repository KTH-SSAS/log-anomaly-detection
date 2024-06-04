import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
MSK_TOKEN = "[MSK]"
OOV_TOKEN = "[OOV]"
CLS_TOKEN = "[CLS]"


def merge_count_fields(field_counts: Dict[str, Dict[str, int]]):
    for field in ["user", "domain", "pc"]:
        new_field: Dict[str, int] = {}
        for direction in ["dst", "src"]:
            old_fieldname = f"{direction}_{field}"
            counts = field_counts[old_fieldname]
            for k, v in counts.items():
                try:
                    new_field[k] += v
                except KeyError:
                    new_field[k] = v
            del field_counts[old_fieldname]
            field_counts[field] = new_field
    return field_counts


class FieldVocab(ABC):

    special_tokens: Dict
    size: int
    num_users: int
    field_names: List[str]

    @abstractmethod
    def __init__(self, vocab: OrderedDict) -> None: ...

    @abstractmethod
    def token2idx(self, token: str, field: int): ...

    @abstractmethod
    def idx2token(self, idx: int, field: int): ...

    @classmethod
    @abstractmethod
    def counts2vocab(cls, counts: Union[dict, Path], cutoff: int): ...


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

    def __init__(self, vocab: OrderedDict) -> None:
        super().__init__(vocab)
        self.vocab = vocab
        self.mask_idx = GlobalVocab.special_tokens[MSK_TOKEN]
        self._eos_idx = GlobalVocab.special_tokens[EOS_TOKEN]
        self._sos_idx = GlobalVocab.special_tokens[SOS_TOKEN]

        self.size = len(self.vocab)

        self.num_special_tokens = len(GlobalVocab.special_tokens)

    def token2idx(self, token, field) -> int:
        try:
            return self.vocab[token]
        except KeyError:
            return self.special_tokens[OOV_TOKEN]

    def idx2token(self, idx, field):
        for k, v in self.vocab.items():
            if idx == v:
                return k
        return OOV_TOKEN

    @classmethod
    def counts2vocab(cls, counts: Union[dict, Path], cutoff: int):
        """Generates a vocabulary file based on a file of token counts per
        field.

        Tokens that appear in several fields are assigned just one
        (global) index.
        """

        vocab: OrderedDict[str, int] = OrderedDict()
        index = 0

        # Add special tokens to entry unrelated to fields
        for k, v in cls.special_tokens.items():
            vocab[k] = v
            index += 1

        if isinstance(counts, Path):
            with open(counts, encoding="utf8") as f:
                field_counts: Dict[str, Dict[str, int]] = json.load(f)
        else:
            field_counts = counts

        # Merge user, domain and pc fields
        field_counts = merge_count_fields(field_counts)

        for field in field_counts:
            # Add indexes for the tokens in the field
            for token, count in field_counts[field].items():
                if count > cutoff and token not in vocab:
                    vocab[token] = index
                    index += 1

        # Add indexes for timestamps
        for i in range(0, 24):
            vocab[f"T{i}"] = index
            index += 1

        print(f"Generated vocab with {index} words.")

        return cls(vocab)

    @property
    def eos_idx(self):
        return self._eos_idx

    @property
    def sos_idx(self):
        return self._sos_idx


class LANLVocab(FieldVocab):
    """Vocabulary that maintains a seperate wordlist for each log field."""

    merge_fields = False

    def __init__(self, vocab: OrderedDict) -> None:
        super().__init__(vocab)
        # Vocab size including special tokens
        self.size = 0
        for token_set in vocab.values():
            self.size += len(token_set)

        self.special_tokens = vocab["special_tokens"]
        del vocab["special_tokens"]

        mask_tokens = vocab[MSK_TOKEN]
        oov_tokens = vocab[OOV_TOKEN]
        del vocab[OOV_TOKEN]
        del vocab[MSK_TOKEN]

        self.field_indexes = {field: i for i, field in enumerate(vocab)}

        # Save field names, without special keys
        self.field_names = list(vocab.keys())

        # Use list to index fields
        self.vocab: List[Dict[str, int]] = list(vocab.values())

        self.mask_tokens = np.zeros(len(self.field_names))
        self.oov_tokens = np.zeros(len(self.field_names))

        for field_name, index in self.field_indexes.items():
            self.mask_tokens[index] = oov_tokens[field_name]
            self.oov_tokens[index] = mask_tokens[field_name]

        # Precalculated vocab limits to help generating random tokens for each field
        self.field_vocab_max = np.zeros(len(self.field_names))
        for i in range(len(self.field_names)):
            self.field_vocab_max[i] = max(self.vocab[i].values())

        self.field_vocab_min = np.zeros(len(self.field_names))
        for i in range(len(self.field_names)):
            self.field_vocab_min[i] = min(self.vocab[i].values())

        self._eos_idx = self.special_tokens[EOS_TOKEN]
        self._sos_idx = self.special_tokens[SOS_TOKEN]

    @property
    def num_users(self):
        """Count the OOV token as a user, but not the mask token."""
        return len(self.vocab[self.field_indexes["src_usr"]]) + 1

    @property
    def eos_idx(self):
        return self._eos_idx

    @property
    def sos_idx(self):
        return self._sos_idx

    def token2idx(self, token: str, field: int) -> int:
        """Returns the index of the given token for a given field."""
        try:
            return self.vocab[field][token]
        except KeyError:
            return self.oov_tokens[field]

    def idx2token(self, idx, field: int) -> str:
        """Returns a token that maps to the given index."""

        for k, i in self.vocab[field].items():
            if i == idx:
                return k

        return OOV_TOKEN

    @classmethod
    def counts2vocab(cls, counts: Union[Dict, Path], cutoff: int):
        """Generates a vocabulary file based on a file of token counts per
        field.

        Tokens are assigned different indexes for each field they appear
        in.
        """

        # Use an ordered dict to maintain the order of field names
        vocab: OrderedDict[str, Dict[str, int]] = OrderedDict()

        # Add special tokens to entry unrelated to fields
        vocab["special_tokens"] = {}
        index = 0
        for t in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, CLS_TOKEN]:
            vocab["special_tokens"][t] = index
            index += 1

        field_counts: Dict[str, Dict[str, int]]
        if isinstance(counts, Path):
            with open(counts, encoding="utf8") as f:
                field_counts = dict(json.load(f))
        else:
            field_counts = counts

        # Merge user, domain and pc fields
        if cls.merge_fields:
            field_counts = merge_count_fields(field_counts)

        # Add one Out-Of-Vocabulary and MASK index for each field
        vocab[MSK_TOKEN] = {}
        vocab[OOV_TOKEN] = {}
        for t in [OOV_TOKEN, MSK_TOKEN]:
            for field in field_counts:
                vocab[t][field] = index
                index += 1

        # Add indexes for timestamps
        vocab[OOV_TOKEN]["time"] = index
        index += 1
        vocab[MSK_TOKEN]["time"] = index
        index += 1
        vocab["time"] = {}
        for i in range(0, 24):
            vocab["time"][f"T{i}"] = index
            index += 1

        for field in field_counts:
            vocab[field] = {}

            # Add indexes for the tokens in the field
            for token, count in field_counts[field].items():
                if count > cutoff:
                    if token in vocab[field]:
                        raise RuntimeError("Duplicate token!")
                    vocab[field][token] = index
                    index += 1

        vocab.move_to_end(MSK_TOKEN)
        vocab.move_to_end(OOV_TOKEN)
        vocab.move_to_end("special_tokens")

        print(f"Generated vocab with {index} words.")

        return cls(vocab)


class MergedLANLVocab(LANLVocab):
    """Vocabulary that maintains a seperate wordlist for each log field, but
    with user, domain and pc merged.

    Uses the following fields: usr, domain, pc, auth_type, logon_type,
    auth_orient, success
    """

    merge_fields = True

    def __init__(self, vocab):
        super().__init__(vocab)
        # List of which fields in the log line correspond to the merged log line
        # not very pedagogical but it werks
        idxs = self.field_indexes
        self.mappings = [
            idxs["time"],
            idxs["user"],
            idxs["domain"],
            idxs["user"],
            idxs["domain"],
            idxs["pc"],
            idxs["pc"],
            idxs["auth_type"],
            idxs["logon_type"],
            idxs["auth_orient"],
            idxs["success"],
        ]

    @property
    def num_users(self):
        """Count the OOV token as a user, but not the mask token."""
        return len(self.vocab[self.field_indexes["user"]]) + 1

    def token2idx(self, token: str, field: int) -> int:
        """Returns the index of the given token for a given field."""

        # Map merged fields like src_user and dst_user to user
        mapped_field = self.mappings[field]
        return super().token2idx(token, mapped_field)

    def idx2token(self, idx, field: int) -> str:
        mapped_field = self.mappings[field]
        return super().idx2token(idx, mapped_field)
