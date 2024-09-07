import torch.nn as nn

from layers.Mamba_Family import Mamba_Layer, AM_Layer
from layers.SelfAttention_Family import FFTBlock
from layers.Embed import DataEmbedding_pos
from Modules.layers import SS1D

class TransMamba_de(nn.Module):
    """
    TransMamba
    """
    def __init__(self, configs):
        super(TransMamba_de, self).__init__()
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
        self.dec_embedding = DataEmbedding_pos(self.d_model, self.dropout)

        self.mamba_preprocess = Mamba_Layer(SS1D(self.d_model, d_state=self.d_state, d_conv=3), self.d_model)
        self.AM_layers = nn.ModuleList(
            [
                AM_Layer(
                    FFTBlock(self.d_model, self.n_heads, self.d_k, self.d_v, self.d_model, self.d_conv, dropout=self.dropout),
                    SS1D(self.d_model, d_state=self.d_state, d_conv=3),
                    self.d_model,
                    self.dropout
                )
                for _ in range(self.d_layers)
            ]
        )
        self.out_proj=nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, encoder_output, mask=None):
        x = self.dec_embedding(encoder_output)
        x = self.mamba_preprocess(x)

        max_len = x.shape[1]
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        for i in range(self.d_layers):
            x = self.AM_layers[i](x, mask=mask, slf_attn_mask=slf_attn_mask)
        out = self.out_proj(x)

        return out