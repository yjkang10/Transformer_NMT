import torch.nn as nn
from layers import DecoderLayer, EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_ff, n_head, n_layers, dropout_p, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_k, d_ff, n_head, dropout_p, device) for _ in range(n_layers)])
    
    def forward(self, src, enc_pad_mask):
        """
        |src| = (batch_size, max_sent)
        |return| = (batch_size, max_sent, d_model)
        """
        for layer in self.layer_stack:
            src = layer(src, enc_pad_mask)
        return self.layer_norm(src)


class Decoder(nn.Module):
    def __init__(self, d_model, d_k, d_ff, n_head, n_layers, dropout_p, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_k, d_ff, n_head, dropout_p, device) for _ in range(n_layers)])

    def forward(self, trg, enc_out, dec_mask, dec_pad_mask):
        """
        |trg| = (batch_size, seq_len)
        |enc_out| = (batch_size, max_sent, d_model)
        |return| = (batch_size, seq_len, d_model)
        """
        for layer in self.layer_stack:
            trg = layer(trg, enc_out, dec_mask, dec_pad_mask)
        return self.layer_norm(trg)
