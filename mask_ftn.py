import torch

def get_next_sent_mask(seq_len, device):
    mask = torch.ones(seq_len, seq_len)
    mask = mask.to(device)
    return 1-torch.tril(mask)

def get_pad_mask(input, device): 
    # input = trg input or src input.
    batch_size, seq_len = input.size()
    mask = torch.zeros_like(input).to(device)
    return torch.eq(input, mask).view(batch_size, 1, 1, seq_len)

def get_trg_mask(input, device):
    next_sent_mask = get_next_sent_mask(input.size(-1), device)
    pad_mask = get_pad_mask(input, device)
    return torch.max(pad_mask, next_sent_mask)
