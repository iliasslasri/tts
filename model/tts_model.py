import torch
import torch.nn as nn
import torch.nn.functional as F

from text_bone.text_encoder import TextEncoder


class TTSModel(nn.Module):
    def __init__(self, text_vocab_size, text_embed_dim, text_num_layers,
                 encodec_codebook_size, encodec_num_quantizers, rvq_embed_dim,
                 num_decoder_layers=1, num_heads=1):
        """
        Args:
            text_vocab_size: size of text tokenizer
            text_embed_dim: embedding dim for text backbone
            text_num_layers: number of transformer layers in text backbone
            encodec_codebook_size: number of discrete codes per quantizer in EnCodec
            encodec_num_quantizers: number of quantizers in EnCodec (e.g., 8)
            rvq_embed_dim: embedding dimension for RVQ decoder (default = text_embed_dim)
            num_decoder_layers: number of transformer decoder layers
            num_heads: number of attention heads
        """
        super().__init__()
        
        # -------------------
        # Text backbone
        # -------------------
        self.text_bone = TextEncoder(
            vocab_size=text_vocab_size,
            embed_dim=text_embed_dim,
            num_layers=text_num_layers
        )

        # Project encoder output to match decoder embedding space
        self.text_to_decoder_proj = nn.Linear(text_embed_dim, rvq_embed_dim)

        # -------------------
        # RVQ decoder backbone
        # -------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=rvq_embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * rvq_embed_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Embedding for RVQ input tokens (for teacher forcing or autoregressive gen)
        self.rvq_token_embed = nn.Embedding(encodec_codebook_size, rvq_embed_dim)

        # -------------------
        # Prediction heads for each quantizer
        # -------------------
        self.quantizers = nn.ModuleList([
            nn.Linear(rvq_embed_dim, encodec_codebook_size)
            for _ in range(encodec_num_quantizers)
        ])
        

    def forward(self, token_ids, rvq_token_ids):
        """
        Args:
            token_ids: [batch, seq_len] input text token IDs
            rvq_token_ids: [B, L_audio] previous EnCodec tokens
        Returns:
            List of predictions per quantizer, each [batch, seq_len, codebook_size]
        """
        # Encoder
        x = self.text_bone(token_ids)  # [B, L, D]
        memory = self.text_to_decoder_proj(x)  # [B, L_text, D]

        # Prepare RVQ input embeddings (shifted right for teacher forcing)
        tgt_emb = self.rvq_token_embed(rvq_token_ids)  # [B, L_audio, D]
        # Generate causal mask
        tgt_len = rvq_token_ids.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=rvq_token_ids.device) * float('-inf'), diagonal=1)

        # Decoder
        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [B, L_audio, D]

        decoded = F.dropout(decoded, p=0.1, training=self.training)

        # Predict code logits for each quantizer
        logits = [q(decoded) for q in self.quantizers]
        return logits
