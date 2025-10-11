import json
import re
from typing import List, Union


class TextTokenizer:
    """
    Basic tokenizer for TTS.
    Can operate in two modes:
        - 'char': character-level tokenization
        - 'phoneme': uses pre-tokenized phoneme sequences
    """

    def __init__(self, vocab_path: str, mode: str = "char"):
        """
        Args:
            vocab_path: optional path to load/save vocabulary
            mode: 'char' or 'phoneme'
        """
        assert mode in ("char", "phoneme"), "mode must be 'char' or 'phoneme'"
        self.mode = mode
        self.vocab = {}
        self.inv_vocab = {}
        if vocab_path:
            self.load_vocab(vocab_path)

    # ----------------------------------------------------
    # Vocabulary utilities
    # ----------------------------------------------------
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from training corpus.
        """
        if self.mode == "char":
            symbols = sorted(set("".join(texts)))
        else:
            # expects space-separated phoneme strings like "HH AH L OW"
            tokens = set()
            for line in texts:
                tokens.update(line.strip().split())
            symbols = sorted(tokens)

        # add special tokens
        symbols = ["<pad>", "<unk>", "<bos>", "<eos>"] + symbols
        self.vocab = {s: i for i, s in enumerate(symbols)}
        self.inv_vocab = {i: s for s, i in self.vocab.items()}
        return self.vocab

    def save_vocab(self, path: str):
        with open(path, "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        with open(path, "r") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {i: s for s, i in self.vocab.items()}

    # ----------------------------------------------------
    # Encoding / Decoding
    # ----------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        """
        if self.mode == "char":
            tokens = list(text.strip())
        else:
            tokens = text.strip().split()

        ids = [self.vocab.get("<bos>")]
        for t in tokens:
            ids.append(self.vocab.get(t, self.vocab["<unk>"]))
        ids.append(self.vocab.get("<eos>"))
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text or phoneme string.
        """
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        tokens = [t for t in tokens if t not in ("<pad>", "<bos>", "<eos>")]
        return "".join(tokens) if self.mode == "char" else " ".join(tokens)

    # ----------------------------------------------------
    # Utility
    # ----------------------------------------------------
    def __len__(self):
        return len(self.vocab)
