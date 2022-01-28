import torch

def get_pos(d_model, max_len):
    enc = torch.FloatTensor(max_len, d_model).zero_()
    pos = torch.arange(0, max_len).unsqueeze(-1).float()  # (max_len, 1)
    dim = torch.arange(0, d_model // 2).unsqueeze(0).float()  # (1, d_model//2)

    enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(d_model)))
    enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(d_model)))

    return enc  # (max_len, d_model)

def get_mask(x, len, max_len):
    mask = []
    for l in len:
        if max_len - l > 0:
            mask = mask + [torch.cat([x.new_ones(1, l).zero_(), x.new_ones(1, (max_len - l))], dim=-1)]
        else:
            # case of max_len == l
            mask = mask + [x.new_ones(1, l).zero_()]

    return torch.cat(mask, dim=0).bool()   # (batch_size, max_len)