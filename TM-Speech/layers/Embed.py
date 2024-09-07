import torch
import torch.nn as nn

import math
from utils import Constants
from text.symbols import symbols


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()

        n_src_vocab = len(symbols) + 1
        self.tokenConv = nn.Embedding(
            n_src_vocab, d_model, padding_idx=Constants.PAD
        )
        torch.nn.init.xavier_uniform_(self.tokenConv.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x

class DataEmbedding_pos(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DataEmbedding_pos, self).__init__()
        self.d_model = d_model
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, self.d_model)
            masked_x = x.masked_fill(expanded_mask, 0)
            pos = self.position_embedding(x)
            x = masked_x + pos
        else:
            pos = self.position_embedding(x)
            x = x + pos
        return self.dropout(x)

class DataEmbedding_value_pos(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DataEmbedding_value_pos, self).__init__()

        self.value_embedding = TokenEmbedding(d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        pos = self.position_embedding(x)
        x = self.value_embedding(x) + pos
        return self.dropout(x)