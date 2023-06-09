import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from .blip2 import Blip2Base, disabled_train
from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig

logger = logging.getLogger(__name__)


class VideoChat_it(Blip2Base):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        # pretrained_path
        vit_model = config.get("vit_model", "eva_clip_g")
        vit_model_path = config.get("vit_model_path", None)
        q_former_model_path = config.get("q_former_model_path", None)
        llama_model_path = config.get("llama_model_path")
        videochat_model_path = config.get("videochat_model_path", "")  
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        img_size = config.get("img_size")
        drop_path_rate = config.get("drop_path_rate", 0)
        use_grad_checkpoint = config.get("use_grad_checkpoint", False)
        vit_precision = config.get("vit_precision", "fp16")
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
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
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        # debug
        debug = config.get("debug", False)

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        self.vit_precision = vit_precision
        logger.info(f'Loading VIT. Use fp16: {vit_precision}')
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
            logger.info("freeze vision encoder")
            if not freeze_mhra:
                open_list = []
                for name, param in self.visual_encoder.named_parameters():
                    if 'mhra' not in name:
                        param.requires_grad = False
                    else:
                        open_list.append(name)
                logger.info(f"open module: {open_list}")
                logger.info("open ln_vision")
            else:
                for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
                self.visual_encoder = self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                for name, param in self.ln_vision.named_parameters():
                    param.requires_grad = False
                self.ln_vision = self.ln_vision.eval()
                self.ln_vision.train = disabled_train
        logger.info('Loading VIT Done')

        logger.info('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(model_path=q_former_model_path)
        logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
        self.extra_query_tokens = nn.Parameter(
            torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
        )

        if freeze_qformer:
            logger.info("freeze Qformer")
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
        logger.info('Loading Q-Former Done')

        logger.info('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if debug:
            logger.info("Debug mode, build small LLAMA")
            llama_config = LlamaConfig.from_pretrained(llama_model_path)
            llama_config.hidden_size = 512
            llama_config.intermediate_size = 2048
            llama_config.num_attention_heads = 8
            llama_config.num_hidden_layers = 12
            llama_config.torch_dtype = torch.float16
            self.llama_model = LlamaForCausalLM(llama_config)
        else:
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

        logger.info("freeze LLAMA")
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len

        # load weights of VideoChat
        if videochat_model_path:
            logger.info(f"Load VideoChat from: {videochat_model_path}")
            ckpt = torch.load(videochat_model_path, map_location="cpu")
            msg = self.load_state_dict(ckpt['model'], strict=False)
            logger.info(msg)

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
            use_image = True if T == 1 else False
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
        return inputs_llama, use_image
        
    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def forward(self, image, text_input):
        img_embeds, use_image = self.encode_img(image)
        batch_size, img_len, _ = img_embeds.shape

        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []
        # handle each prompt individually
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.begin_signal + self.role[0] + ": "
            sep2 = self.begin_signal + self.role[1] + ": "
            raw_text = p_after.split(sep2)
            for idx in range(1, len(raw_text)):
                raw_text[idx] = sep2 + raw_text[idx]
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # target: "###Human:       ###Assistant: xxxxx. ###"
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system.rstrip())
            sep_len = self._get_text_len(sep1.rstrip())
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            assert cur_len == answer_targets.shape[1], f"The final length is not equal to the original prompt: {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
        
        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device) * self.llama_tokenizer.pad_token_id
        inputs_embeds = self.llama_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.llama_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len+1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_len+1):(input_len+1)] = target_list[idx][0, :(input_len-p_before_len-img_len)]

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
    
        return dict(
            loss=outputs.loss,
        )
