import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.transmamba import TransMamba
from Modules.transmamba_de import TransMamba_de
from Modules.transformer import Transformer
from Modules.decoder import Decoder
from Modules.postnet import PostNet
from Modules.varianceadaptor import VarianceAdaptor
from utils.tools import get_mask_from_lengths

class TM_Speech(nn.Module):
    """ TransformerMamba_Speech """

    def __init__(self, preprocess_config, model_config):
        super(TM_Speech, self).__init__()
        self.model_config = model_config
        if model_config["AM"]['encoder_type'] == 'Transformer':
            self.encoder = Transformer(model_config)
        elif model_config["AM"]['encoder_type'] == 'TransMamba':
            self.encoder = TransMamba(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = TransMamba_de(model_config)
        # self.model = VSSM(preprocess_config, model_config)
        self.mel_linear = nn.Linear(
            model_config["decoder"]["d_model"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.mel_conv = nn.Conv1d(in_channels=model_config["decoder"]["d_model"], out_channels=80, kernel_size=1)
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["AM"]["d_model"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)


        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        # output = self.decoder(output)
        output = self.decoder(output, mask=mel_masks)
        output = output.permute(0,2,1)
        output = self.mel_conv(output)
        output = output.permute(0, 2, 1)
        # output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens
        )