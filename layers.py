import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        """
        |q| = (batch_size, m, d_model)
        |k, v| = (batch_size, n, d_model)
        |mask| = (batch_size, m, n)
        |output| = (batch_size, m, d_model)
        """
        attn = torch.matmul(q, k.transpose(-1, -2))
        # |attn| = (batch_size, m, n)
        if mask is not None:
            # attn += mask * (-1e9)
            attn.masked_fill_(mask, -float('inf'))
        attn = self.softmax(attn / (self.d_k ** 0.5))
        return torch.bmm(attn, v)  #

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k

        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

        self.attention = Attention(d_k)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, q, k, v, mask=None):
        """
        |q| = (batch_size, m, d_model)
        |k, v| = (batch_size, n, d_model)
        |mask| = (batch_size, m, n)
        |output| = (batch_size, m, d_model)
        """
        qw = self.linear_q(q).split(self.d_model // self.n_heads, dim=-1)
        kw = self.linear_k(k).split(self.d_model // self.n_heads, dim=-1)
        vw = self.linear_v(v).split(self.d_model // self.n_heads, dim=-1)
        # |qw_i| = (batch_size, m, d_model/n_heads)
        # |kw_i, vw_i| = (batch_size, n, d_model/n_heads)

        qw = torch.cat(qw, dim=0)
        kw = torch.cat(kw, dim=0)
        vw = torch.cat(vw, dim=0)
        # |qw| = (batch_size * n_heads, m, d_model/n_heads)
        # |kw, vw| = (batch_size * n_heads, n, d_model/n_heads)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_heads)], dim=0)
            # |mask| = (batch_size * n_heads, m, n)

        attn = self.attention(qw, kw, vw, mask)  
        # (batch_size * n_heads, m, d_model/n_heads)
        attn = attn.split(q.size(0), dim=0) 
        # |attn_i| = (batch_size, m, d_model/n_heads)
        return self.linear(torch.cat(attn, dim=-1)) 

class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_ff, n_heads, dropout_p):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, d_k, n_heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.ffnn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=False),
            nn.Linear(d_ff, d_model))
        
        self.ffnn_norm = nn.LayerNorm(d_model)
        self.ffnn_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        """
        |x| = (batch_siz, n, d_model)
        |mask| = (batch_size, n, n)
        |return| = (batch_size, n, d_model)
        """
        z = self.attn_norm(x)
        z = x + self.attn_dropout(self.multihead_attn(z, z, z, mask))
        return z + self.ffnn_dropout(self.ffnn(self.ffnn_norm(z))), mask
    
class Decoder(nn.Module):
    def __init__(self, d_model, d_k, d_ff, n_heads, dropout_p):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, d_k, n_heads)
        self.masked_attn_norm = nn.LayerNorm(d_model)
        self.masked_attn_dropout = nn.Dropout(dropout_p)

        self.attn = MultiHeadAttention(d_model, d_k, n_heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.ffnn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model))
        
        self.ffnn_norm = nn.LayerNorm(d_model)
        self.ffnn_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_val, mask, prev, future_mask):
        """
        |key_val| = (batch_size, n, d_model)
        |mask| = (batch_size, m, n)
        """
        if prev is None:  # training
            # |x| = (batch_size, m, d_model)
            # |future_mask| = (batch_size, m, m)
            # |z| = (batch_size, m, d_model)
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(self.masked_attn(z, z, z, future_mask))

        else:  # evaluation
            # |x| = (batch_size, 1, d_model)
            # |prev| = (batch_size, t-1, d_model)
            # |future_mask| = None
            # |z| = (batch_size, 1, d_model)
            prev = self.masked_attn_norm(prev)
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(self.masked_attn(z, prev, prev, mask=None))
             
        key_val = self.attn_norm(key_val)
        z = z + self.attn_dropout(self.attn(self.attn_norm(z), key_val, key_val, mask))
        z = z + self.ffnn_dropout(self.ffnn(self.ffnn_norm(z)))

        return z, key_val, mask, prev, future_mask
        

class NewSequential(nn.Sequential):
    def forward(self, *x):
        """
        nn.Sequential doesn't provide mutiple input arguments and returns.
        """
        for module in self._modules.values():
            x = module(*x)
        return x 
    

