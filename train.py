import math
import os
import re
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import WhisperTokenizer
from whisper_normalizer.english import EnglishTextNormalizer

from datasets import LJSpeechDataset, collate_fn
from encodec.encodec.encodec import EncodecModel
from model.tts_model import TTSModel
from torch.optim import Optimizer

BASE_DIR = Path(__file__).resolve().parent

# Taken from HF transformers
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig = None):
    print(OmegaConf.to_yaml(cfg))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    writer = SummaryWriter(log_dir=f"{cfg.logging.log_dir}/tts_{timestamp}")
    checkpoint_dir = Path(f"{cfg.logging.log_dir}/tts_{timestamp}/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and EnCodec
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
    tokenizer.set_prefix_tokens(language="en", task="transcribe")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.to(device)

    # Dataset & DataLoader
    wav_dir = BASE_DIR / Path(cfg.dataset.root_dir) / cfg.dataset.wav_dir
    text_file = BASE_DIR / Path(cfg.dataset.root_dir) / cfg.dataset.text_file

    # Parse the transcripts
    pattern = re.compile(r'^\(\s*(\S+)\s+"(.+)"\s*\)$')
    rows = []
    english_normalizer = EnglishTextNormalizer()

    with open(text_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                file_id, text = match.groups()
                wav_path = wav_dir / f"{file_id}.wav"
                wav_path = os.path.abspath(os.path.join(BASE_DIR, wav_path))
                rows.append(
                    {
                        "file_id": file_id,
                        "text": english_normalizer(text),
                        "path": str(wav_path),
                    }
                )

    df = pd.DataFrame(rows)
    print(f"All audio files: {len(df)}")
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(
        f"Valid audio files: {len(df)} remaining, after filtering with os.path.exists"
    )
    print("Head of dataFrame \n")
    print(df.head())
    dataset = LJSpeechDataset(
        df,
        tokenizer,
        encodec_model,
        sample_rate=cfg.train.sample_rate,
        num_samples=cfg.train.num_samples,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Model
    text_vocab_size = tokenizer.vocab_size

    model = TTSModel(
        text_vocab_size,
        text_embed_dim=cfg.model.text_embed_dim,
        text_num_layers=cfg.model.text_num_layers,
        encodec_codebook_size=cfg.rvq.n_bins,
        encodec_num_quantizers=cfg.rvq.n_quantizers,
        rvq_embed_dim=cfg.model.text_embed_dim,
        num_decoder_layers=cfg.model.num_decoder_layers,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    # lr tuning
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.n_epochs, eta_min=1e-6)
    effective_training_steps = (len(dataloader) * cfg.train.n_epochs) // cfg.train.accumulation_steps
    num_warmup_steps = int(0.25 * effective_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=effective_training_steps,
    )

    print("-" * 100)
    print("SUMMARY OF MODEL")
    print("-" * 100)
    token_ids_example = torch.randint(
        0, tokenizer.vocab_size, (1, 11), dtype=torch.long
    )
    rvq_token_ids_example = torch.randint(
        0, cfg.rvq.n_bins, (1, 300, cfg.rvq.n_quantizers), dtype=torch.long
    )

    # Move to device if needed
    token_ids_example = token_ids_example.to(device)
    rvq_token_ids_example = rvq_token_ids_example.to(device)

    print(model)
    print("Total params:", sum(p.numel() for p in model.parameters()))
    print(
        "Trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("-" * 100)
    print("-" * 100)
    # Training loop
    for epoch in range(cfg.train.n_epochs):
        model.train()
        total_loss = 0
        batch_idx = 0
        for token_ids, encodec_batch in dataloader:
            global_step = epoch * len(dataloader) + batch_idx
            batch_idx += 1
            token_ids = token_ids.to(device)
            encodec_batch = [q.to(device) for q in encodec_batch]

            optimizer.zero_grad()
            assert (
                token_ids.max() < tokenizer.vocab_size
            ), "Token ID exceeds vocab size!"
            # encodec_batch: list of [B, L] per quantizer, length = N_QUANTIZERS
            # Stack along last dimension: [B, L, N_QUANTIZERS]
            rvq_targets = torch.stack(encodec_batch, dim=-1)  # [B, L, N_QUANTIZERS]

            # For teacher forcing, shift right and add a <BOS> token
            rvq_input = torch.full(
                (rvq_targets.shape[0], 1, cfg.rvq.n_quantizers),
                cfg.train.bos_token,
                dtype=torch.long,
                device=device,
            )

            # Concatenate BOS with all tokens except last
            rvq_token_ids = torch.cat(
                [rvq_input, rvq_targets[:, :-1, :]], dim=1
            )  # [B, L, N_QUANTIZERS]

            logits = model(
                token_ids, rvq_token_ids
            )  # list of [B, L, codebook_size] per quantizer

            # compute cross-entropy for each quantizer
            pred = logits
            target = encodec_batch

            # Ensure seq lengths match
            target_stack = torch.stack(target, dim=2)
            min_len = min(pred.shape[1], target_stack.shape[-2])
            pred = pred[:, :min_len, :, :]
            target_stack = target_stack[..., :min_len, :]
            logits = pred.permute(0, 3, 1, 2)  # [B, L, Q, N_BINS] -> [B, N_BINS, L, Q]
            loss = F.cross_entropy(logits, target_stack)

            # Only update weights every accumulation_steps
            # Scale the loss
            loss = loss / cfg.train.accumulation_steps
            loss.backward()
            del rvq_targets, rvq_token_ids
            del pred, target_stack
            # print loss every batch
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
            print("gloabl step", global_step, "loss=", loss.item())
            if (global_step + 1) % cfg.train.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            # optimizer.step()
            # scheduler.step()
            total_loss += loss.item()

            # reconstruct audio from predicted tokens for monitoring
            if global_step % cfg.logging.save_every == 0:
                with torch.no_grad():
                    # take argmax as predicted tokens for now (TODO)
                    pred_tokens = torch.argmax(logits, dim=-1)
                    # pred_tokens: list of [B, L] per quantizer
                    pred_list = [
                        pred_tokens[..., i] for i in range(cfg.rvq.n_quantizers)
                    ]
                    pred_tokens_tensor = torch.stack(pred_list, dim=1).to("cpu")
                    encoded_frames = [(pred_tokens_tensor, None)]
                    encodec_model_cpu = encodec_model.to("cpu")
                    reconstructed = encodec_model_cpu.decode(encoded_frames)
                    # reconstructed: [B, 1, T]
                    sf.write(
                        f"runs/tts_{timestamp}/reconstructed_epoch{epoch}.wav",
                        reconstructed[0][0].cpu().numpy(),
                        cfg.train.sample_rate,
                    )

                    gt_tokens = torch.stack(encodec_batch, dim=1).to(
                        "cpu"
                    )  # [B, N_Q, L]
                    encoded_frames_gt = [(gt_tokens, None)]
                    reconstructed_gt = encodec_model_cpu.decode(
                        encoded_frames_gt
                    )  # [B, 1, T]
                    sf.write(
                        f"runs/tts_{timestamp}/original_epoch{epoch}.wav",
                        reconstructed_gt[0][0].cpu().numpy(),
                        cfg.train.sample_rate,
                    )

                    torch.save(
                        model.state_dict(), checkpoint_dir / f"model_epoch_{epoch}.pt"
                    )
                    print(f"[INFO] Saved checkpoint: model_epoch_{epoch}.pt")
                    encodec_model.to(device)
            del logits, encodec_batch, token_ids
    final_model_path = checkpoint_dir / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")
    writer.close()


if __name__ == "__main__":
    main()
