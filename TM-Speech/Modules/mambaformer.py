import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Family import Mamba_Layer
from layers.SelfAttention_Family import FFTBlock
from layers.Embed import DataEmbedding_value_pos
import numpy as np
from mamba.Layers import SS1D

class MambaFormer(nn.Module):
    """
    MambaFormer
    """
    def __init__(self, configs):
        super(MambaFormer, self).__init__()
        self.dec_in = configs["AM"]["dec_in"]
        self.d_model = configs["AM"]["d_model"]
        self.dropout = configs["AM"]["dropout"]
        self.factor = configs["AM"]["factor"]
        self.n_heads = configs["AM"]["n_heads"]
        self.d_state = configs["AM"]["d_state"]
        self.d_conv = configs["AM"]["d_conv"]
        self.d_layers = configs["AM"]["d_layers"]
        self.c_out = configs["AM"]["c_out"]
        self.output_attention = configs["AM"]["output_attention"]
        self.d_k = self.d_v = (
                configs["AM"]["d_model"]
                // configs["AM"]["n_heads"]
        )

        # Embedding
        self.dec_embedding = DataEmbedding_value_pos(self.d_model, self.dropout)

        self.mamba_preprocess = Mamba_Layer(SS1D(self.d_model, d_state=self.d_state, d_conv=3), self.d_model)
        self.AT_layers = nn.ModuleList()
        for _ in range(self.d_layers):
            layer = FFTBlock(self.d_model, self.n_heads, self.d_k, self.d_v, self.dec_in, self.d_conv, dropout=self.dropout)
            self.AT_layers.append(layer)
        self.out_proj=nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, src_seq, mask=None):
        x = self.dec_embedding(src_seq)
        x = self.mamba_preprocess(x)

        max_len = x.shape[1]
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        for i in range(self.d_layers):
            x, _ = self.AT_layers[i](x, mask=mask, slf_attn_mask=slf_attn_mask)
        out = self.out_proj(x)

        return out