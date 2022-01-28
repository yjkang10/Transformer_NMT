import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import get_mask, get_pos


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_k, d_ff, n_layers, dropout_p, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads 
        self.d_k = d_k
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_len = max_len
        
        self.emb_src = nn.Embedding(vocab_size, d_model)
        self.emb_trg = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout_p)

        self.pos_enc = get_pos(d_model, max_len)  # (max_len, d_model)
        
        self.encoder = NewSequential(*[Encoder(d_model, d_k, d_ff, n_heads, dropout_p) for _ in range(n_layers)])
        self.decoder = NewSequential(*[Decoder(d_model, d_k, d_ff, n_heads, dropout_p) for _ in range(n_layers)])

        self.generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
            nn.LogSoftmax(dim=-1),
        )


    def _positional_encoding(self, x, init_pos=0):
        """
        |x| = (batch_size, n, d_model)
        |self.pos_enc| = (max_len, d_model)
        """
        assert x.size(-1) == self.pos_enc.size(-1)  # d_model
        assert x.size(1) + init_pos <= self.max_len

        pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)
        # |pos_enc| = (1, n, d_model)
        return x + pos_enc.to(x.device)

    
    def forward(self, x, y):
        """
        |x[0]| = (batch_size, n)
        |y| = (batch_size, m)
        """
        # generate padding mask
        with torch.no_grad():
            mask = get_mask(x[0], x[1], self.max_len)  # (batch_size, n)
            x = x[0]
            mask_src = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
            # |mask_src| = (batch_size, n, n)
            mask_trg = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))
            # |mask_trg| = (batch_size, m, n)
        
        # Encoder
        z = self.emb_dropout(self._positional_encoding(self.emb_src(x)))
        z, _ = self.encoder(z, mask_src)

        # generate future mask
        with torch.no_grad():  
            future_mask = torch.triu(x.new_ones((y.size(1), y.size(1))), diagonal=1).bool()
            # |future_mask| = (m, m)
            future_mask = future_mask.unsqueeze(0).expand(y.size(0), *future_mask.size())
            # |future_mask| = (batch_size, m, m)  

        h = self.emb_dropout(self._positional_encoding(self.emb_trg(y)))
        h, _, _, _, _ = self.decoder(h, z, mask_trg, None, future_mask)
        # |h| = (batch_size, m, hidden_size)
        return self.generator(h)  # (batch_size, m, output_size)
