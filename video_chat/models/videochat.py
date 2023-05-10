import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from .blip2 import Blip2Base, disabled_train
from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig


class VideoChat(Blip2Base):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()

        vit_model = config.get("vit_model", "eva_clip_g")
        vit_model_path = config.get("vit_model_path", None)
        q_former_model_path = config.get("q_former_model_path", None)
        llama_model_path = config.get("llama_model_path")
        videochat_model_path = config.get("videochat_model_path", "")  
        img_size = config.get("img_size")

        drop_path_rate = config.get("drop_path_rate", 0)
        use_grad_checkpoint = config.get("use_grad_checkpoint", False)
        vit_precision = config.get("vit_precision", "fp16")
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        max_txt_len = config.get("max_txt_len", 32)

        # uniformerv2
        freeze_mhra = config.get("freeze_mhra", False)
        temporal_downsample = config.get("temporal_downsample", True)
        no_lmhra = config.get("no_lmhra", False)
        double_lmhra = config.get("double_lmhra", False)
        lmhra_reduction = config.get("lmhra_reduction", 2.0)
        gmhra_layers = config.get("gmhra_layers", 8)
        gmhra_drop_path_rate = config.get("gmhra_drop_path_rate", 0.)
        gmhra_dropout = config.get("gmhra_dropout", 0.5)
        # qformer
        num_query_token = config.get("num_query_token")
        extra_num_query_token = config.get("extra_num_query_token", 64)

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        self.vit_precision = vit_precision
        print(f'Loading VIT. Use fp16: {vit_precision}')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, 
            use_grad_checkpoint, vit_precision, vit_model_path,
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra, 
            double_lmhra=double_lmhra,
            lmhra_reduction=lmhra_reduction, 
            gmhra_layers=gmhra_layers, 
            gmhra_drop_path_rate=gmhra_drop_path_rate,
            gmhra_dropout=gmhra_dropout, 
        )
        if freeze_vit:
            print("freeze vision encoder")
            if not freeze_mhra:
                open_list = []
                for name, param in self.visual_encoder.named_parameters():
                    if 'mhra' not in name:
                        param.requires_grad = False
                    else:
                        open_list.append(name)
                print(f"open module: {open_list}")
                print("open ln_vision")
            else:
                for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
                self.visual_encoder = self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                for name, param in self.ln_vision.named_parameters():
                    param.requires_grad = False
                self.ln_vision = self.ln_vision.eval()
                self.ln_vision.train = disabled_train
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features,
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(model_path=q_former_model_path)
        print(f"Add extra {extra_num_query_token} tokens in QFormer")
        self.extra_query_tokens = nn.Parameter(
            torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
        )

        if freeze_qformer:
            print("freeze Qformer")
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )

        print("freeze LLAMA")
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len

        # load weights of VideoChat
        if videochat_model_path:
            print(f"Load VideoChat from: {videochat_model_path}")
            ckpt = torch.load(videochat_model_path, map_location="cpu")
            msg = self.load_state_dict(ckpt['model'], strict=False)
            print(msg)

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            T = image.shape[1]
            # use_image = True if T == 1 else False
            image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama
