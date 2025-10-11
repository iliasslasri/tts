import torch
import torch.utils.data as Dataset
import torchaudio


class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, tokenizer, encodec, sample_rate=24000):
        """
        metadata_path: LJSpeech metadata CSV/TSV with format: id|text|file_path
        audio_dir: path to audio files
        tokenizer: Whisper tokenizer
        encodec: pretrained EnCodec encoder
        """
        self.samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    self.samples.append((parts[1], parts[2]))  # text, file_path
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.encodec = encodec
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, file_path = self.samples[idx]
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Load audio
        wav, sr = torchaudio.load(f"{self.audio_dir}/{file_path}")
        wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0, keepdim=True)  # convert to mono if stereo with mean of all channels
        wav = wav / wav.abs().max()  # normalize to -1 to 1

        # Encode audio to EnCodec discrete tokens
        with torch.no_grad():
            encodec_out = self.encodec.encode(wav)  
            # encodec_out: list of [B, seq_len] per quantizer
        return token_ids, encodec_out
    
# ------------------------
# Utils
# ------------------------
def collate_fn(batch):
    # batch: list of tuples (token_ids, encodec_out)
    token_ids_list, encodec_list = zip(*batch)

    # pad token_ids
    token_ids_lens = [len(x) for x in token_ids_list]
    max_len = max(token_ids_lens)
    token_ids_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(token_ids_list):
        token_ids_batch[i, :len(x)] = x

    # pad encodec per quantizer
    num_quantizers = len(encodec_list[0])
    encodec_batch = []
    for q in range(num_quantizers):
        seqs = [e[q].squeeze(0) for e in encodec_list]  # remove channel
        max_seq = max([len(s) for s in seqs])
        padded = torch.zeros(len(batch), max_seq, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
        encodec_batch.append(padded)
    
    return token_ids_batch, encodec_batch