import torch.nn as nn
from layers import EmbedLayer
from module import Decoder, Encoder
from mask_ftn import get_pad_mask, get_trg_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_sent, dropout_p, n_layers, d_model, d_k, d_ff, n_head, device, pad_idx=0):
        super().__init__()
        self.device = device
        self.embedding_layer = EmbedLayer(vocab_size, embedding_dim, max_sent, dropout_p, device, pad_idx=0)
        self.encoder = Encoder(d_model, d_k, d_ff, n_head, n_layers, dropout_p, device)
        self.decoder = Decoder(d_model, d_k, d_ff, n_head, n_layers, dropout_p, device)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight_ = self.embedding_layer.embedding.weight.T

    def forward(self, src, trg):
        enc_pad_mask = get_pad_mask(src, self.device)
        enc_out = self.embedding_layer(src)
        enc_out = self.encoder(enc_out, enc_pad_mask)
        dec_mask = get_trg_mask(trg, self.device)
        dec_pad_mask = get_pad_mask(trg, self.device)
        dec_out = self.embedding_layer(trg)
        dec_out = self.decoder(dec_out, enc_out, dec_mask, dec_pad_mask)
        enc_pad_mask.detach()
        dec_pad_mask.detach()
        dec_mask.detach()
        del enc_pad_mask, dec_pad_mask, dec_mask
        return self.fc(dec_out)