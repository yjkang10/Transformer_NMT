import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_sent, embedding_dim, device):
        """
        positional encoding
        embedding_dim = d_model
        """
        super().__init__()
        self.pos_encoding = torch.empty(max_sent, embedding_dim)
        for pos in range(max_sent):
            for i in range(embedding_dim//2):
                div_term = pos / (10000**(2*i/embedding_dim))
                div_term = torch.FloatTensor([div_term])
                self.pos_encoding[pos][2*i] = torch.sin(div_term)
                self.pos_encoding[pos][2*i+1] = torch.cos(div_term)
        self.pos_encoding = self.pos_encoding.to(device)

    def forward(self, x):
        """
        |x| = (batch_size, max_sent, embedding_dim)
        """
        return x + self.pos_encoding

class EmbedLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_sent, dropout_p, device, pad_idx=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)   
        self.position = PositionalEncoding(max_sent, embedding_dim, device)
        self.dropout = nn.Dropout(dropout_p)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        """
        input |x| = (batch_size, max_sent)
        output |x| = (batch_size, max_sent, embedding_dim)
        """
        x = self.embedding(x) * (self.embedding_dim ** 0.5)
        x = self.position(x)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, d_k, dropout_p, device):
        super().__init__()
        self.d_k = d_k
        self.device = device
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, q, k, v, mask=None):
        """
        |q, k| = (batch_size, n_head, seq_len, d_k)
        |v| = (batch_size, n_head, seq_len, d_v)
        |mask| = (batch_size, 1, 1(or seq_len), seq_len)
        """
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / (self.d_k ** 0.5)
        if mask is not None:
            attn += mask * (-1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_head, dropout_p, device):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(d_k, dropout_p, device)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.att = None
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, q, k, v, mask=None):
        """
        |q, k, v| = (batch_size, seq_len, embedding_dim)
        |trg_mask| = (batch_size, 1, 1(or seq_len), seq_len)
        |out| = (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = q.size()
        q = self.linear_q(q).view(batch_size, seq_len, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, seq_len, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, seq_len, self.n_head, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn, self.att = self.attention(q, k, v, mask)  # (batch_size, n_head, seq_len, d_v)
        attn = attn.transpose(1, 2)
        attn = attn.contiguous().view(batch_size, seq_len, -1)  # (batch_size, seq_len, d_model)
        return self.linear(attn)

class PositionwiseFFNN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(d_model)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        """
        |x| = (batch_size, seq_len, d_model)
        """
        # residual = x.clone()
        # x = self.relu(self.linear1(x))
        # x = self.linear2(self.dropout(x))
        # x += residual
        # return self.layer_norm(x)
        residual = x.clone()
        x = self.layer_norm(x)
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        return self.dropout(x) + residual


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_ff, n_head, dropout_p, device):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, d_k, n_head, dropout_p, device)
        self.pos_ffnn = PositionwiseFFNN(d_model, d_ff, dropout_p)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, enc_pad_mask):
        """
        |x| = (batch_siz, max_sent, d_model)
        |enc_mask| = (batch_size, 1, 1, seq_len)
        |return| = (batch_size, seq_len, d_model)
        """
        residual = x.clone()
        x = self.layer_norm(x)
        x = self.multihead_attn(x, x, x, enc_pad_mask)
        x = self.dropout(x) + residual
        return self.pos_ffnn(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_ff, n_head, dropout_p, device):
        super().__init__()
        self.masked_multihead_attn = MultiHeadAttention(d_model, d_k, n_head, dropout_p, device)
        self.attn_to_encoder = MultiHeadAttention(d_model, d_k, n_head, dropout_p, device)
        self.pos_ffnn = PositionwiseFFNN(d_model, d_ff, dropout_p)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, enc_out, dec_mask, dec_pad_mask):
        """
        |x| = (batch_size, seq_len, d_model)
        |enc_out| = (batch_size, max_sent, d_model)
        |dec_mask| = (batch_size, 1, seq_len, seq_len)
        |dec_pad_mask| = (batch_size, 1, 1, seq_len)
        |return| = (batch_size, seq_len, d_model)
        """
        residual = x.clone()
        x = self.layer_norm(x)
        x = self.masked_multihead_attn(x, x, x, dec_mask)
        x = self.dropout(x) + residual

        residual_2 = x.clone()
        x = self.layer_norm(x)
        x = self.attn_to_encoder(x, enc_out, enc_out, dec_pad_mask)
        x = self.dropout(x) + residual_2
        return self.pos_ffnn(x)