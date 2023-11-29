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
from .vit import build_vit
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class Blip2Base(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side, local_files_only=True)
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
        qformer_hidden_dropout_prob=0.1,
        qformer_attention_probs_dropout_prob=0.1,
        qformer_drop_path_rate=0.,
    ):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased", local_files_only=True)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        encoder_config.hidden_dropout_prob = qformer_hidden_dropout_prob
        encoder_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob
        encoder_config.drop_path_list = [x.item() for x in torch.linspace(0, qformer_drop_path_rate, encoder_config.num_hidden_layers)]
        logger.info(f"Drop_path:{encoder_config.drop_path_list}")
        logger.info(encoder_config)
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    @classmethod
    def init_vision_encoder_umt(self, config):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        vision_encoder = build_vit(config)

        if config.vision_encoder.vit_add_ln:
            vision_layernorm = nn.LayerNorm(config.vision_encoder.encoder_embed_dim, eps=1e-12)
        else:
            vision_layernorm = nn.Identity()

        return vision_encoder, vision_layernorm


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
