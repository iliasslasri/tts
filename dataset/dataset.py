import torch
import torch.utils.data as Dataset
import torchaudio
from datasets import load_dataset


class LJSpeechDataset(Dataset):
    def __init__(self, dataset_path_hf, tokenizer, encodec, sample_rate=24000, num_samples=100):
        """
        dataset: path to dataset on huggingface
        tokenizer: Whisper tokenizer
        encodec: pretrained EnCodec encoder
        sample_rate: target sample rate for audio, will resample if different
        num_samples: number of samples to load from the dataset
        """
        self.samples = []
        dataset = load_dataset(dataset_path_hf, split="train")
        self.tokenizer = tokenizer
        self.encodec = encodec
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        for i, item in enumerate(dataset):
            text = item["text"]
            audio_path = item["audio"]["path"]
            self.samples.append((text, audio_path))
            if i + 1 >= num_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, file_path = self.samples[idx]
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Load audio
        file_path = self.samples[idx][1]
        wav, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
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