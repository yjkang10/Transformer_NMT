from torch import log_softmax
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from search import SingleBeamSearchBoard


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

        self.pos_enc = self._get_pos(d_model, max_len)  # (max_len, d_model)
        
        self.encoder = NewSequential(*[Encoder(d_model, d_k, d_ff, n_heads, dropout_p) for _ in range(n_layers)])
        self.decoder = NewSequential(*[Decoder(d_model, d_k, d_ff, n_heads, dropout_p) for _ in range(n_layers)])

        self.generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    @torch.no_grad()
    def _get_pos(self, d_model, max_len):
        enc = torch.FloatTensor(max_len, d_model).zero_()
        pos = torch.arange(0, max_len).unsqueeze(-1).float()  # (max_len, 1)
        dim = torch.arange(0, d_model // 2).unsqueeze(0).float()  # (1, d_model//2)

        enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(d_model)))
        enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(d_model)))

        return enc  # (max_len, d_model)

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

    @torch.no_grad()
    def _get_mask(self, x, len):
        mask = []
        max_len = max(len)
        for l in len:
            if max_len - l > 0:
                mask += [torch.cat([x.new_ones(1, l).zero_(), x.new_ones(1, (max_len - l))], dim=-1)]
            else:
                # case of max_len == l
                mask += [x.new_ones(1, l).zero_()]
        
        return torch.cat(mask, dim=0).bool()   # (batch_size, max_len)
    
    def forward(self, x, y):
        """
        |x[0]| = (batch_size, n)
        |y| = (batch_size, m)
        """
        # generate padding mask
        with torch.no_grad():
            mask = self._get_mask(x[0], x[1])  # (batch_size, n)
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
        h, _, _, _, _ = self.decoder(h, key_val=z, mask=mask_trg, prev=None, future_mask=future_mask)
        # |h| = (batch_size, m, hidden_size)
        return self.generator(h)  # (batch_size, m, output_size)

    def beam_search(self, x, beam_size=7, max_len=255, n_best=1, len_penalty=.2):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)
        n_layers = len(self.decoder._modules)
        mask = self._get_mask(x[0], x[1])  # (batch_size, n)
        x = x[0]

        mask_src = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        # |mask_src| = (batch_size, n, n)
        mask_trg = mask.unsqueeze(1)
        # |mask_trg| = (batch_size, 1, n)

        z = self.emb_dropout(self._positional_encoding(self.emb_src(x)))
        z, _ = self.encoder(z, mask_src)
        # |z| = (batch_size, n, hidden_size)

        prev_status = {}
        for layer_idx in range(n_layers + 1):
            prev_status[f'prev_state_{layer_idx}'] = {'init_status': None, 'batch_dim_dix': 0}
        
        boards = [
            SingleBeamSearchBoard(
                z.device,
                prev_status,
                beam_size=beam_size,
                max_len=max_len,
            ) for _ in range(batch_size)
        ]
        done_cnt = [board.is_done() for board in boards]

        length = 0
        while sum(done_cnt) < batch_size and length <= max_len:
            fab_input, fab_z, fab_mask = [], [], []
            fab_prevs = [[] for _ in range(n_layers + 1)]

            for i, board in enumerate(boards): # sample in minibatch
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()

                    fab_input += [y_hat_i]
                    fab_z += [z[i].unsqueeze(0)] * beam_size
                    fab_mask += [mask_trg[i].unsqueeze(0)] * beam_size

                    for layer_idx in range(n_layers + 1):
                        prev_i = prev_status[f'prev_state_{layer_idx}']
                        if prev_i is not None:
                            fab_prevs[layer_idx] += [prev_i]
                        else:
                            fab_prevs[layer_idx] = None

            fab_input = torch.cat(fab_input, dim=0)
            fab_z     = torch.cat(fab_z,     dim=0)
            fab_mask  = torch.cat(fab_mask,  dim=0)
            for i, fab_prev in enumerate(fab_prevs): # i == layer_idx
                if fab_prev is not None:
                    fab_prevs[i] = torch.cat(fab_prev, dim=0)
            # |fab_input|    = (current_batch_size, 1,)
            # |fab_z|        = (current_batch_size, n, d_model)
            # |fab_mask|     = (current_batch_size, 1, n)
            # |fab_prevs[i]| = (current_batch_size, length, d_model)
            # len(fab_prevs) = n_layers + 1

            h_t = self.emb_dropout(self._positional_encoding(self.emb_trg(fab_input), init_pos=length))
            # |h_t| = (current_batch_size, 1, d_model)
            if fab_prevs[0] is None:
                fab_prevs[0] = h_t
            else:
                fab_prevs[0] = torch.cat([fab_prevs[0], h_t], dim=1)

            for layer_idx, decoder_block in enumerate(self.decoder._modules.values()):
                prev = fab_prevs[layer_idx]
                # |prev| = (current_batch_size, m, d_model)

                h_t, _, _, _, _ = decoder_block(h_t, fab_z, fab_mask, prev, None)
                # |h_t| = (current_batch_size, 1, d_model)

                if fab_prevs[layer_idx + 1] is None:
                    fab_prevs[layer_idx + 1] = h_t
                else:
                    fab_prevs[layer_idx + 1] = torch.cat([fab_prevs[layer_idx + 1], h_t],dim=1) 

            y_hat_t = self.generator(h_t)
            # |y_hat_t| = (batch_size, 1, vocab_size)

            # |fab_prevs[i][begin:end]| = (beam_size, length, d_model)
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    begin = cnt * beam_size
                    end = begin + beam_size

                    prev_status = {}
                    for layer_idx in range(n_layers + 1):
                        prev_status[f'prev_state_{layer_idx}'] = fab_prevs[layer_idx][begin:end]

                    board.collect_result(y_hat_t[begin:end], prev_status)

                    cnt += 1

            done_cnt = [board.is_done() for board in boards]
            length += 1

        batch_sentences, batch_probs = [], []

        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, len_penalty=len_penalty)

            batch_sentences += [sentences]
            batch_probs += [probs]

        return batch_sentences, batch_probs
