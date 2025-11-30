from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=1, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.transformer(x)
        return x
