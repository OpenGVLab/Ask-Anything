"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import os
import logging

import torch
import torch.nn as nn

from .Qformer import BertConfig, BertLMHeadModel
from .eva_vit import create_eva_vit_g
from transformers import BertTokenizer


class Blip2Base(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(
        cls, 
        num_query_token, vision_width, 
        qformer_hidden_dropout_prob=0.,
        qformer_attention_probs_dropout_prob=0.,
        qformer_drop_path_rate=0.,
    ):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        encoder_config.hidden_dropout_prob = qformer_hidden_dropout_prob
        encoder_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob
        encoder_config.drop_path_list = [x.item() for x in torch.linspace(0, qformer_drop_path_rate, encoder_config.num_hidden_layers)]
        print(f"Drop_path:{encoder_config.drop_path_list}")
        print(encoder_config)
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, 
        model_name, img_size, drop_path_rate, 
        use_grad_checkpoint, precision, vit_model_path,
        temporal_downsample=True,
        no_lmhra=False, 
        double_lmhra=False,
        lmhra_reduction=2.0, 
        gmhra_layers=8, 
        gmhra_drop_path_rate=0.,
        gmhra_dropout=0.5, 
    ):
        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of VideoChat"
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, 
            use_grad_checkpoint, precision, vit_model_path,
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra, 
            double_lmhra=double_lmhra,
            lmhra_reduction=lmhra_reduction, 
            gmhra_layers=gmhra_layers, 
            gmhra_drop_path_rate=gmhra_drop_path_rate,
            gmhra_dropout=gmhra_dropout, 
        )

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def load_from_pretrained(self, model_path):
        if model_path is not None and os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        print(f"Load QFormer from {model_path}")
        print(msg)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
