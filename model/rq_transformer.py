import torch
from copy import deepcopy

class RQTransformerDecoder(torch.nn.Module):
    def __init__(self, decoder_layer, num_layers, encodec_num_quantizers):
        super().__init__()
        self.temporal_decoder = torch.nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers//2)]
        )
        self.num_layers = num_layers
        self.depth_decoder = torch.nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers//2)]
        )
        self.encodec_num_quantizers = encodec_num_quantizers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        batch_size, seq_len, _ = tgt.size()
        # wheighed sum on the sequence length dimension of memory
        tgt = tgt.view(batch_size, seq_len//self.encodec_num_quantizers, self.encodec_num_quantizers, -1)  # [B, L, NQ, D]
        tgt = tgt.mean(dim=2, keepdim=True).squeeze(2) # [B, L, D]
        
        # Temporal decoding
        zs = tgt
        L = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(L, L, device=tgt.device), diagonal=1).bool()
        for layer in self.temporal_decoder:
            # temporal context
            zs = layer(
                zs,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # Depth decoding for each quantizer
        outputs = []
        L = zs.size(1)
        rqv_memory = torch.zeros(batch_size, L, zs.size(-1), device=tgt.device)
        tgt_mask = None  # No causal mask for depth decoding
        for _ in range(self.encodec_num_quantizers):
            for layer in self.depth_decoder:
                out = layer(
                    zs + rqv_memory,
                    rqv_memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            rqv_memory = out
            outputs.append(out)
        # Concatenate outputs along the feature dimension
        final_output = torch.cat(outputs, dim=1)  # [B, L*NQ, D]  
        return final_output


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 4
    NQ = 3
    d_model = 8
    num_layers = 4

    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=1, batch_first=True)
    rq_transformer = RQTransformerDecoder(decoder_layer, num_layers=num_layers, encodec_num_quantizers=NQ)

    tgt = torch.randn(batch_size, seq_len, NQ, d_model)
    memory_mask = torch.tril(torch.ones(seq_len, seq_len)) == 1 
    memory = torch.randn(batch_size, seq_len, d_model)

    out = rq_transformer(tgt, memory)
    print("Output shape:", out.shape)  # Expected: [batch_size, seq_len, NQ, d_model]
    import ipdb; ipdb.set_trace()