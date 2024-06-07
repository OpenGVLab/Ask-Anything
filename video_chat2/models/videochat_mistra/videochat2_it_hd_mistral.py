import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType

from ..blip2.blip2 import Blip2Base, disabled_train
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class VideoChat2_it_hd_mistral(Blip2Base):
    """
    VideoChat2 model.
    """
    def __init__(self, config):
        super().__init__()
        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        mistral_model_path = config.get("mistral_model_path")
        videochat2_model_path = config.get("videochat2_model_path", "")  
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        freeze_llm = config.get("freeze_llm", True)
        # vit
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # qformer
        num_query_token = config.get("num_query_token")
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        self.qformer_text_input = config.get("qformer_text_input", False)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        logger.info(f"Add instruction in qformer: {self.qformer_text_input}")
        # debug
        self.debug = config.get("debug", False)
        self.llm_bf16 = config.get("llm_bf16", False)
        use_flash_attention = config.get("use_flash_attention", False)
        self.use_lora = config.get("use_lora", False)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)
        # dynamic resolution
        self.local_size = config.dynamic_config.get("local_size", 224)
        self.add_global = config.dynamic_config.get("add_global", True)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.tokenizer.padding_side = "left"
        self.low_resource = low_resource
        self.vision_encoder, self.vision_layernorm = self.init_vision_encoder_umt(config)
        self.qformer, self.query_tokens = self.init_Qformer(
            num_query_token, config.vision_encoder.encoder_embed_dim,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )
        
        if not self.qformer_text_input:
            self.qformer.bert.embeddings.word_embeddings = None
            self.qformer.bert.embeddings.position_embeddings = None
            for layer in self.qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.qformer.resize_token_embeddings(len(self.tokenizer))
        self.qformer.cls = None

        if vit_blip_model_path:
            logger.info(f"Load ViT and QFormer from {vit_blip_model_path}")
            state_dict = torch.load(vit_blip_model_path, map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info('Loading ViT and Q-Former Done')    
        
        self.extra_num_query_token = extra_num_query_token
        if extra_num_query_token > 0:
            logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
            self.extra_query_tokens = nn.Parameter(
                torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
            )

        if freeze_vit:
            logger.info("freeze vision encoder")
            for _, param in self.vision_encoder.named_parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
            for _, param in self.vision_layernorm.named_parameters():
                param.requires_grad = False
            self.vision_layernorm = self.vision_layernorm.eval()
            self.vision_layernorm.train = disabled_train

        if freeze_qformer:
            logger.info("freeze Qformer")
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer = self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        logger.info('Loading Mistral')
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
        self.mistral_tokenizer.padding_side = "left"
        if not self.mistral_tokenizer.pad_token:
            logger.info("Set pad_token")
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

        if self.debug:
            logger.info("Debug mode, build small Mistral")
            mistral_config = AutoConfig.from_pretrained(mistral_model_path)
            mistral_config.hidden_size = 512
            mistral_config.intermediate_size = 2048
            mistral_config.num_attention_heads = 8
            mistral_config.num_hidden_layers = 12
            mistral_config.torch_dtype = torch.float16
            self.mistral_model = AutoModelForCausalLM.from_config(mistral_config)
        else:
            if use_flash_attention:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.bfloat16 if self.llm_bf16 else torch.float16,
                    # use_flash_attention_2=True,
                    attn_implementation="flash_attention_2",
                )
            else:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.bfloat16 if self.llm_bf16 else torch.float16,
                )

        if freeze_llm:
            logger.info("freeze Mistral")
            for _, param in self.mistral_model.named_parameters():
                param.requires_grad = False
        logger.info('Loading Mistral Done')

        if self.use_lora:
            logger.info("Use lora")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj", "lm_head"]
            )
            self.mistral_model = get_peft_model(self.mistral_model, peft_config)
            if not freeze_llm:
                logger.info("Unfreeze Mistral")
                for _, param in self.mistral_model.base_model.named_parameters():
                    param.requires_grad = True
            self.mistral_model.print_trainable_parameters()

        self.mistral_proj = nn.Linear(
            self.qformer.config.hidden_size, self.mistral_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len

        # load weights of VideoChat2
        if videochat2_model_path:
            logger.info(f"Load VideoChat2 from: {videochat2_model_path}")
            ckpt = torch.load(videochat2_model_path, map_location="cpu")
            if 'model' in ckpt.keys():
                msg = self.load_state_dict(ckpt['model'], strict=False)
            else:
                msg = self.load_state_dict(ckpt, strict=False)
            logger.info(msg)

    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def encode_img(self, image, instruction):
        device = image[0].device
        if self.low_resource:
            self.vit_to_cpu()
            image = [img.to("cpu") for img in image]

        with self.maybe_autocast():
            # split the image or video according to the shape
            shapes = []
            input_imgs = []
            input_instructions = []
            for idx, img in enumerate(image):
                # logger.info(f"Input shape: {img.shape}")
                T, C, H, W = img.shape
                shapes.append([H//self.local_size, W//self.local_size])
                sub_img = img.reshape(
                    1, T, 3, H//self.local_size, self.local_size, W//self.local_size, self.local_size
                ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T, 3, self.local_size, self.local_size).contiguous()
                input_imgs.append(sub_img)
                input_instructions.extend([instruction[idx]] * len(sub_img))
                if self.add_global:
                    glb_img = F.interpolate(
                        img.float(), size=(self.local_size, self.local_size), mode='bicubic', align_corners=False
                    ).to(sub_img.dtype)
                    input_imgs.append(glb_img.unsqueeze(0))
                    input_instructions.append(instruction[idx])
            input_imgs = torch.cat(input_imgs, dim=0)

            T = input_imgs.shape[1]
            use_image = True if T == 1 else False
            input_imgs = input_imgs.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

            image_embeds = self.vision_encoder(input_imgs, use_image)
            B, T, L, C = image_embeds.shape
            image_embeds = image_embeds.reshape(B, -1, C)
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
            
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if self.extra_num_query_token > 0:
                query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            else:
                query_tokens = self.query_tokens
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    input_instructions,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image_embeds.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                query_output = self.qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            qformer_features = self.mistral_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            q_C = qformer_features.shape[-1]

            # merge the features from different split
            # stolen from https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b/blob/main/build_mlp.py#L97-L115
            output_imgs = []
            output_len = []
            for [h, w] in shapes:
                B_ = h * w
                if self.add_global:
                    output_imgs.append(qformer_features[:B_+1].view(1, -1, q_C))
                    qformer_features = qformer_features[B_+1:]
                else:
                    output_imgs.append(qformer_features[:B_].view(1, -1, q_C))
                    qformer_features = qformer_features[B_:]
                # logger.info(f"Features shape: {output_imgs[-1].shape}")
                output_len.append(output_imgs[-1].shape[1])

        return output_imgs, output_len, use_image
        
    def _get_text_len(self, text):
        return self.mistral_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def forward(self, image, text_input, instruction):
        if len(image[0].shape) == 1:
            use_text = True
            device = image[0].device
            batch_size = len(image)
            img_lens = [0] * batch_size
        else:
            use_text = False
            img_embeds, img_lens, use_image = self.encode_img(image, instruction)
            device = img_embeds[0].device
            batch_size = len(img_embeds)

        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []
        # handle each prompt individually
        for idx, prompt in enumerate(text_input):
            if use_text:
                p_after = prompt
                p_after_tokens = self.mistral_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)
                if self.use_lora:
                    p_after_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
                else:
                    p_after_embeds = self.mistral_model.model.embed_tokens(p_after_tokens.input_ids)
                input_embeds = p_after_embeds
            else:
                tmp_img_embeds = img_embeds[idx]
                # split the prompt via END_TOKEN
                end_token = self.img_end_token if use_image else self.end_token
                p_before, p_after = prompt.split(end_token)
                p_after = end_token + p_after
                p_before_tokens = self.mistral_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
                p_after_tokens = self.mistral_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
                if self.use_lora:
                    p_before_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                    p_after_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
                else:
                    p_before_embeds = self.mistral_model.model.embed_tokens(p_before_tokens.input_ids)
                    p_after_embeds = self.mistral_model.model.embed_tokens(p_after_tokens.input_ids)
                input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.human_start + " "
            sep2 = " " + self.human_end + " "
            raw_text = p_after.split(sep2)
            for idx in range(0, len(raw_text) - 1):
                raw_text[idx] = raw_text[idx] + sep2
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # [target] "xxxxx. </s>"
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())

            if self.debug:  # Inspect and check the correctness of masking
                z = answer_targets[0].clone()
                z = torch.where(z == -100, self.mistral_tokenizer.unk_token_id, z)
                logger.info(self.mistral_tokenizer.decode(z))
                
            assert cur_len == answer_targets.shape[1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            if use_text:
                p_before_len_list.append(0)
            else:
                p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
   
        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + max(img_lens))
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(device) * self.mistral_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.mistral_model.base_model.model.model.embed_tokens(inputs_embeds)
        else:
            inputs_embeds = self.mistral_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.mistral_tokenizer.bos_token_id
        
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len+1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_lens[idx]+1):(input_len+1)] = target_list[idx][0, :(input_len-p_before_len-img_lens[idx])]

        with self.maybe_autocast():
            outputs = self.mistral_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False, # current flash_attn2 dows not support padding=right for mistral
            )
    
        return dict(
            loss=outputs.loss,
        )
