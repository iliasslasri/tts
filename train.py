# train.py
import re
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from transformers import WhisperTokenizer
from whisper_normalizer.english import EnglishTextNormalizer

from datasets import LJSpeechDataset, collate_fn
from encodec.encodec.encodec import EncodecModel
from model.tts_model import TTSModel
from pathlib import Path

NUM_SAMPLES = 1
N_EPOCHS = int(10e4)
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 24_000

# RVQ
N_BINS = 1024
N_QUANTIZERS = 32
BATCH_SZ = 1

def main():
    writer = SummaryWriter(log_dir=f"runs/tts_{int(time.time())}")
    device = "cpu" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and EnCodec
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
    tokenizer.set_prefix_tokens(language="en", task="transcribe")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.to(device)

    # Dataset & DataLoader
    root = Path("./datasets/cmu_us_awb_arctic")
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
    print("-"*100)
    print("SUMMARY OF MODEL")
    print("-"*100)
    token_ids_example = torch.randint(0, tokenizer.vocab_size, (1, 11), dtype=torch.long)
    rvq_token_ids_example = torch.randint(0, N_BINS, (1, 300, N_QUANTIZERS), dtype=torch.long)

    # Move to device if needed
    token_ids_example = token_ids_example.to(device)
    rvq_token_ids_example = rvq_token_ids_example.to(device)

    summary(model, input_data=(token_ids_example, rvq_token_ids_example))

    print(model)
    print("Total params:", sum(p.numel() for p in model.parameters()))
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("-"*100)
    print("-"*100)

    # Training loop
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        batch_idx = 0
        for token_ids, encodec_batch in dataloader:
            global_step = epoch * len(dataloader) + batch_idx
            batch_idx += 1 
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
            pred = logits
            target = encodec_batch

            # Ensure lengths match
            target_stack = torch.stack(target, dim=2) 
            L_pred, L_target = pred.shape[1], target_stack.shape[-2]
            min_len = min(L_pred, L_target)
            pred = pred[:, :min_len, :, :]
            target_stack = target_stack[..., :min_len, :]
            target_onehot = F.one_hot(target_stack, num_classes=N_BINS).float()
            loss = F.cross_entropy(pred, target_onehot)

            loss.backward()

            # print loss every batch
            writer.add_scalar("Loss/train", loss.item(), global_step)

            optimizer.step()
            total_loss += loss.item()

            # reconstruct audio from predicted tokens for monitoring
            if epoch % 1 == 0:
                with torch.no_grad():
                    # take argmax as predicted tokens for now (TODO)
                    pred_tokens = torch.argmax(logits, dim=-1)
                    # pred_tokens: list of [B, L] per quantizer
                    pred_list = [pred_tokens[..., i] for i in range(N_QUANTIZERS)]
                    pred_tokens_tensor = torch.stack(pred_list, dim=1)
                    encoded_frames = [(pred_tokens_tensor, None)]
                    reconstructed = encodec_model.decode(encoded_frames)
                    # reconstructed: [B, 1, T]
                    torchaudio.save(f"reconstructed_epoch{epoch}.wav", reconstructed[0].cpu(), SAMPLE_RATE)

        print(f"Epoch {epoch} - Loss: {total_loss / len(dataloader)}")
    writer.close()
if __name__ == "__main__":
    main()
