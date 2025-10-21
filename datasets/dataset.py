import os

import torch
import torchaudio
from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    def __init__(self, dataset, tokenizer, encodec, sample_rate=24000, num_samples=100):
        """
        dataset: pandas DataFrame
        tokenizer: Whisper tokenizer
        encodec: pretrained EnCodec encoder
        sample_rate: target sample rate for audio, will resample if different
        num_samples: number of samples to load from the dataset
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.encodec = encodec
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        for i in range(len(dataset)):
            text = dataset.loc[i, "text"]
            audio_path = dataset.loc[i,"path"]
            self.samples.append((text, audio_path))
            if i + 1 >= num_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, file_path = self.samples[idx]
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Load audio
        wav, sr = safe_load_wav(file_path)
        if wav is None:
            return self.__getitem__(idx+1)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0, keepdim=True)  # convert to mono if stereo with mean of all channels
        wav = wav / wav.abs().max()  # normalize to -1 to 1
        # Encode audio to EnCodec discrete tokens
        device = next(self.encodec.parameters()).device
        wav = wav.unsqueeze(1).to(device)
        with torch.no_grad():
            encodec_out = self.encodec.encode(wav)  
            # encodec_out: list of [B, seq_len] per quantizer
        return token_ids, encodec_out
    
# ------------------------
# Utils
# ------------------------
def collate_fn(batch):
    """
    Collate function to pad sequences in the batch
    Args:
        batch: list of tuples (token_ids, encodec_out)
    Returns:
        token_ids_batch: [batch, max_seq_len] padded token IDs
        encodec_batch: list of [batch, max_seq_len] padded encodec tokens per quantizer
    """

    # batch: list of tuples (token_ids, encodec_out)
    token_ids_list, encodec_list = zip(*batch)

    # pad token_ids
    token_ids_lens = [len(x) for x in token_ids_list]
    max_len = max(token_ids_lens)
    token_ids_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(token_ids_list):
        token_ids_batch[i, :len(x)] = x

    # pad encodec per quantizer
    num_quantizers = encodec_list[0][0][0].shape[1]
    encodec_batch = []
    # import ipdb
    # ipdb.set_trace()
    for q in range(num_quantizers):
        # Extract the q-th quantizer sequence for each sample
        seqs = [e[0][0][0][q].cpu() for e in encodec_list]  # shape: [seq_len]

        max_seq = max(s.size(0) for s in seqs)

        # Pad each sequence to max length
        padded = torch.zeros(len(seqs), max_seq, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s

        encodec_batch.append(padded)
    
    return token_ids_batch, encodec_batch

def safe_load_wav(path, sample_rate=24000):
    try:
        wav, sr = torchaudio.load(path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        return wav, sample_rate
    except Exception as e:
        print(f"[WARN] Skipping bad file: {path} ({e})")
        return None, None