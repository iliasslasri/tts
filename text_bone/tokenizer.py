from typing import List, Union

import torch
from transformers import WhisperTokenizer


class TextTokenizer:
    """Tokenizer wrapper for TTS.

    Can use:
        - 'char' mode (original)
        - 'whisper' mode (BPE from OpenAI Whisper)
    """

    def __init__(self, mode="char", vocab_path=None):
        assert mode in ("char", "whisper"), "mode must be 'char' or 'whisper'"
        self.mode = mode

        if self.mode == "whisper":
            # load pretrained Whisper tokenizer
            self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")  # nosec
        else:
            self.vocab = {}
            self.inv_vocab = {}
            if vocab_path:
                self.load_vocab(vocab_path)

    # ------------------------
    # Char-mode methods
    # ------------------------
    def build_vocab(self, texts: List[str]):
        if self.mode != "char":
            raise NotImplementedError("build_vocab only works in char mode")
        symbols = sorted(set("".join(texts)))
        symbols = ["<pad>", "<unk>", "<bos>", "<eos>"] + symbols
        self.vocab = {s: i for i, s in enumerate(symbols)}
        self.inv_vocab = {i: s for s, i in self.vocab.items()}
        return self.vocab

    def save_vocab(self, path: str):
        if self.mode != "char":
            raise NotImplementedError("save_vocab only works in char mode")
        import json

        with open(path, "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        if self.mode != "char":
            raise NotImplementedError("load_vocab only works in char mode")
        import json

        with open(path) as f:
            self.vocab = json.load(f)
        self.inv_vocab = {i: s for s, i in self.vocab.items()}

    # ------------------------
    # Encoding / Decoding
    # ------------------------
    def encode(self, text: str) -> List[int]:
        if self.mode == "whisper":
            # encode text to BPE token IDs
            return self.tokenizer(text).input_ids
        else:
            tokens = list(text.strip())
            ids = [self.vocab.get("<bos>")]
            for t in tokens:
                ids.append(self.vocab.get(t, self.vocab["<unk>"]))
            ids.append(self.vocab.get("<eos>"))
            return ids

    def decode(self, ids: List[int]) -> str:
        if self.mode == "whisper":
            # decode BPE token IDs back to text
            return self.tokenizer.decode(ids)
        else:
            tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
            tokens = [t for t in tokens if t not in ("<pad>", "<bos>", "<eos>")]
            return "".join(tokens)

    # ------------------------
    # Utility
    # ------------------------
    def __len__(self):
        if self.mode == "whisper":
            return self.tokenizer.vocab_size
        return len(self.vocab)


class SimpleCharTokenizer:
    """A simple character-level tokenizer.

    Vocab includes: Special tokens, A-Z, a-z, numbers, and punctuation found in the dataset.
    """

    def __init__(self):
        # Special Tokens
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

        self.chars = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789" " .,?!'-:;\"()"
        )

        self.vocab = self.special_tokens + list(self.chars)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self.token_to_id["<pad>"]

    @property
    def bos_token_id(self):
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self):
        return self.token_to_id["<eos>"]

    @property
    def unk_token_id(self):
        return self.token_to_id["<unk>"]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Converts string to list of IDs."""
        ids = []
        for c in text:
            # Fallback to <unk> if character not in vocab
            ids.append(self.token_to_id.get(c, self.unk_token_id))
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        """Converts IDs back to string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        chars = []
        for i in ids:
            # Skip special tokens during decoding for readability
            if i >= len(self.special_tokens):
                chars.append(self.id_to_token.get(i, ""))
        return "".join(chars)

    # Mock method to satisfy potential calls relying on HF interface
    def set_prefix_tokens(self, language=None, task=None):
        pass
