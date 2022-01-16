import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    def __init__(self, label_smoothing, vocab_size, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        smoothing = label_smoothing / (vocab_size - 2)    # word itself, and pad token
        one_hot = torch.full((vocab_size,), smoothing)
        one_hot[self.pad_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))   # register buffer is not a parameter, but in state_dict.
        self.confidence = 1.0 - label_smoothing

    def forward(self, model_out, gold):
        """
        model_out : batch_size * n_classes
        gold : batch_size
        """
        model_prob = self.one_hot.repeat(gold.size(0), 1)                
        model_prob.scatter_(1, gold.unsqueeze(1), self.confidence)
        mask = (gold == self.pad_idx)
        model_prob.masked_fill_(mask.unsqueeze(1), 0)      # broadcasting
        pred = model_out.log_softmax(dim=-1)
        return torch.sum(-pred*model_prob) / sum(gold != self.pad_idx)
