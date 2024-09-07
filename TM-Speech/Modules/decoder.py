import torch.nn as nn

from layers.SelfAttention_Family import FFTBlock
from layers.Embed import DataEmbedding_value_pos, DataEmbedding_pos

import numpy as np

class Decoder(nn.Module):
    """
    Decoder Only Transformer
    """
    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.dec_in = configs["decoder"]["dec_in"]
        self.d_model = configs["decoder"]["d_model"]
        self.conv_filter_size = configs["decoder"]["conv_filter_size"]
        self.dropout = configs["decoder"]["dropout"]
        self.factor = configs["decoder"]["factor"]
        self.n_heads = configs["decoder"]["n_heads"]
        self.d_conv = configs["decoder"]["d_conv"]
        self.d_layers = configs["decoder"]["d_layers"]
        self.d_ff = configs["decoder"]["d_ff"]
        self.activation = configs["decoder"]["activation"]
        self.c_out = configs["decoder"]["c_out"]
        self.output_attention = configs["decoder"]["output_attention"]
        self.d_k = self.d_v = (
                configs["decoder"]["d_model"]
                // configs["decoder"]["n_heads"]
        )

        # Embedding
        self.dec_embedding = DataEmbedding_pos(self.d_model, self.dropout)

        self.decoder_layers = nn.ModuleList(
            [
                FFTBlock(self.d_model, self.n_heads, self.d_k, self.d_v, self.conv_filter_size, self.d_conv, dropout=self.dropout)
                for _ in range(self.d_layers)
            ]
        )
        self.out_proj=nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, x_dec, mask=None):
        x = self.dec_embedding(x_dec, mask=mask)
        max_len = x.shape[1]
        # -- Prepare masks
        dec_self_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        for i in range(self.d_layers):
            x, _ = self.decoder_layers[i](x, mask=mask, slf_attn_mask=dec_self_mask)
        out = self.out_proj(x)

        return out