import torch
import torch.nn as nn
import torch.nn.functional as F

from text_bone.text_encoder import TextEncoder


class TTSModel(nn.Module):
    def __init__(self, text_vocab_size, text_embed_dim, text_num_layers,
                 encodec_codebook_size, encodec_num_quantizers):
        """
        Args:
            text_vocab_size: size of text tokenizer
            text_embed_dim: embedding dim for text backbone
            text_num_layers: number of transformer layers in text backbone
            encodec_codebook_size: number of discrete codes per quantizer in EnCodec
            encodec_num_quantizers: number of quantizers in EnCodec (e.g., 8)
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

        # intermediate text representation
        # TODO add non-linearity? and more layers?
        self.text_projection = nn.Linear(text_embed_dim, text_embed_dim)

        # -------------------
        # TODO Add decoder backbone for auto-regressive generation
        # -------------------

        # Prediction head for EnCodec tokens
        # -------------------
        # Predict each quantizer separately
        # TODO: try joint prediction
        self.quantizers = nn.ModuleList([
            nn.Linear(text_embed_dim, encodec_codebook_size) 
            for _ in range(encodec_num_quantizers)
        ])
        

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch, seq_len] input text token IDs
        Returns:
            List of predictions per quantizer, each [batch, seq_len, codebook_size]
        """
        x = self.text_bone(token_ids)  # [B, L, D]
        x = self.text_projection(x)    # [B, L, D]
        x = F.relu(x)
        # x: [B, L, D]
        x = F.dropout(x, p=0.1, training=self.training)

        # predict codes for each quantizer
        logits = [q(x) for q in self.quantizers]  # list of [B, L, codebook_size]
        return logits
