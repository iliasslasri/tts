# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer

from encodec import EncodecModel

from .dataset import LJSpeechDataset, collate_fn
from .model.tts_model import TTSModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and EnCodec
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.eval()
    encodec_model.to(device)

    # Dataset & DataLoader
    dataset = LJSpeechDataset(
        metadata_path="metadata.csv",
        audio_dir="wavs",
        tokenizer=tokenizer,
        encodec=encodec_model
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Model
    text_vocab_size = tokenizer.vocab_size
    text_embed_dim = 512
    text_num_layers = 6
    encodec_codebook_size = 1024
    encodec_num_quantizers = 8

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
        
        print(f"Epoch {epoch} - Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    main()
