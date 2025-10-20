# train.py
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from datasets import LJSpeechDataset, collate_fn
from encodec.encodec.encodec import EncodecModel
from model.tts_model import TTSModel

NUM_SAMPLES = 10e3
BASE_DIR = "/home/iliass/tts/"
SAMPLE_RATE = 24_000

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and EnCodec
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.set_prefix_tokens(language="en", task="transcribe")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.eval()
    encodec_model.to(device)

    # Dataset & DataLoader
    root = Path("./cmu_us_awb_arctic")
    wav_dir = root / "wav"
    text_file = root / "etc" / "txt.done.data"

    # Parse the transcripts
    pattern = re.compile(r'^\(\s*(\S+)\s+"(.+)"\s*\)$')
    rows = []

    with open(text_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                file_id, text = match.groups()
                import os
                wav_path = wav_dir / f"{file_id}.wav"
                wav_path = os.path.abspath(os.path.join(BASE_DIR, wav_path))
                print(wav_path)
                rows.append({"file_id": file_id, "text": text, "path": str(wav_path)})

    df = pd.DataFrame(rows)
    print(f"All audio files: {len(df)}")
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Valid audio files: {len(df)} remaining, after filtering with os.path.exists")
    print("Head of dataFrame \n")
    print(df.head())
    dataset = LJSpeechDataset(df, tokenizer, encodec_model, sample_rate=SAMPLE_RATE, num_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Model
    text_vocab_size = tokenizer.vocab_size
    text_embed_dim = 512
    text_num_layers = 6
    encodec_codebook_size = 1024
    encodec_num_quantizers = 32

    model = TTSModel(
        text_vocab_size, text_embed_dim, text_num_layers,
        encodec_codebook_size, encodec_num_quantizers
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
            logits = model(token_ids)  # list of [B, L, codebook_size] per quantizer

            # compute cross-entropy for each quantizer
            loss = 0
            for i in range(encodec_num_quantizers):
                # [B, L, C] -> [B, C, L] for F.cross_entropy
                pred = logits[i].transpose(1, 2)
                target = encodec_batch[i]
                loss += F.cross_entropy(pred, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # print loss every batch
            print(f"Batch Loss: {loss.item()}")

            # reconstruct audio from predicted tokens for monitoring
            if epoch % 1 == 0:
                with torch.no_grad():
                    # take argmax as predicted tokens for now (TODO)
                    pred_tokens = [torch.argmax(logit, dim=-1) for logit in logits]
                    # pred_tokens: list of [B, L] per quantizer
                    reconstructed = encodec_model.decode(pred_tokens)
                    # reconstructed: [B, 1, T]
                    torchaudio.save(f"reconstructed_epoch{epoch}.wav", reconstructed.cpu(), 24000)

        print(f"Epoch {epoch} - Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    main()
