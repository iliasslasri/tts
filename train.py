# train.py
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import whisper
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer
from whisper_normalizer.english import EnglishTextNormalizer

from datasets import LJSpeechDataset, collate_fn
from encodec.encodec.encodec import EncodecModel
from model.tts_model import TTSModel

NUM_SAMPLES = 10
BASE_DIR = "/home/iliass/tts/"
SAMPLE_RATE = 24_000

# RVQ
N_BINS = 1024
N_QUANTIZERS = 32
BATCH_SZ = 1

def main():
    device = "cpu" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and EnCodec
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
    tokenizer.set_prefix_tokens(language="en", task="transcribe")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.to(device)

    # Dataset & DataLoader
    root = Path("./cmu_us_awb_arctic")
    wav_dir = root / "wav"
    text_file = root / "etc" / "txt.done.data"

    # Parse the transcripts
    pattern = re.compile(r'^\(\s*(\S+)\s+"(.+)"\s*\)$')
    rows = []
    english_normalizer = EnglishTextNormalizer()
    
    with open(text_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                file_id, text = match.groups()
                import os
                wav_path = wav_dir / f"{file_id}.wav"
                wav_path = os.path.abspath(os.path.join(BASE_DIR, wav_path))
                rows.append({"file_id": file_id, "text": english_normalizer(text), "path": str(wav_path)})

    df = pd.DataFrame(rows)
    print(f"All audio files: {len(df)}")
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Valid audio files: {len(df)} remaining, after filtering with os.path.exists")
    print("Head of dataFrame \n")
    print(df.head())
    dataset = LJSpeechDataset(df, tokenizer, encodec_model, sample_rate=SAMPLE_RATE, num_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SZ, shuffle=True, collate_fn=collate_fn)

    # Model
    text_vocab_size = tokenizer.vocab_size
    text_embed_dim = 512
    text_num_layers = 6

    model = TTSModel(
        text_vocab_size, text_embed_dim, text_num_layers,
        N_BINS, N_QUANTIZERS, rvq_embed_dim=text_embed_dim
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for token_ids, encodec_batch in dataloader:
            token_ids = token_ids.to(device)
            encodec_batch = [q.to(device) for q in encodec_batch]

            optimizer.zero_grad()
            assert token_ids.max() < tokenizer.vocab_size, "Token ID exceeds vocab size!"
            # encodec_batch: list of [B, L] per quantizer, length = N_QUANTIZERS
            # Stack along last dimension: [B, L, N_QUANTIZERS]
            rvq_targets = torch.stack(encodec_batch, dim=-1)  # [B, L, N_QUANTIZERS]

            # For teacher forcing, shift right and add a <BOS> token 
            bos_token = 0
            rvq_input = torch.full((rvq_targets.shape[0], 1, N_QUANTIZERS), bos_token, dtype=torch.long, device=device)

            # Concatenate BOS with all tokens except last
            rvq_token_ids = torch.cat([rvq_input, rvq_targets[:, :-1, :]], dim=1)  # [B, L, N_QUANTIZERS]
            
            logits = model(token_ids, rvq_token_ids)  # list of [B, L, codebook_size] per quantizer

            # compute cross-entropy for each quantizer
            # loss = 0
            # for i in range(N_BINS):
            pred = logits
            target = encodec_batch

            # import ipdb
            # ipdb.set_trace()
            # Ensure lengths match
            target_stack = torch.stack(target, dim=2) 
            L_pred, L_target = pred.shape[1], target_stack.shape[-2]
            min_len = min(L_pred, L_target)
            pred = pred[:, :min_len, :, :]
            target_stack = target_stack[..., :min_len, :]
            target_onehot = F.one_hot(target_stack, num_classes=N_BINS).float()
            loss = F.cross_entropy(pred, target_onehot)
        
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # print loss every batch
            print(f"Batch Loss: {loss.item()}")

            # reconstruct audio from predicted tokens for monitoring
            if epoch % 1 == 0:
                with torch.no_grad():
                    # import ipdb
                    # ipdb.set_trace()
                    # take argmax as predicted tokens for now (TODO)
                    pred_tokens = torch.argmax(logits, dim=-1)
                    # pred_tokens: list of [B, L] per quantizer
                    pred_list = [pred_tokens[..., i] for i in range(N_QUANTIZERS)]
                    pred_tokens_tensor = torch.stack(pred_list, dim=0)
                    encoded_frames = [(pred_tokens_tensor, None)]
                    reconstructed = encodec_model.decode(encoded_frames)
                    # reconstructed: [B, 1, T]
                    torchaudio.save(f"reconstructed_epoch{epoch}.wav", reconstructed.cpu(), 24000)

        print(f"Epoch {epoch} - Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    main()
