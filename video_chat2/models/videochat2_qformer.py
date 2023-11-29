import logging

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from .blip2.vit import build_vit
from .blip2.builder import build_qformer
from .criterions import VTC_VTM_Loss, get_sim
from timm.models.layers import trunc_normal_

logger = logging.getLogger(__name__)


class VideoChat2_qformer(nn.Module):
    """
    VideoChat2 model.
    """
    def __init__(self, config, tokenizer):
        super(VideoChat2_qformer, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        self.vision_width = config.model.vision_encoder.d_model
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim
        self.agg_method = config.model.get("agg_method", "mean")

        if self.config.criterion.get('vtm_add_text_cls', False):
            logger.info('Use text [CLS] for matching: ADD')
        elif self.config.criterion.get('vtm_cat_text_cls', False):
            logger.info('Use text [CLS] for matching: CAT')

        # create modules. seperate vision_encoder and vision_temp_embed as
        # we wish to freeze vision_encoder
        (
            self.vision_encoder,
            self.vision_layernorm,
            self.vision_temp_embed,
        ) = self.build_vision_encoder()
        if self.config.model.get("freeze_vision_encoder", True):
            self.vision_encoder = self.freeze_module(self.vision_encoder)

        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)

        self.qformer, self.query_tokens = build_qformer(
            config.model.qformer_num_query_tokens, self.vision_width,
            config.model.get('qformer_hidden_dropout_prob', 0.1),
            config.model.get('qformer_attention_probs_dropout_prob', 0.1),
            config.model.get('drop_path_rate', 0.),
        )
        self.qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.qformer.state_dict()
        for name, param in self.qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.qformer.config.hidden_size, self.embed_dim)
        self.text_proj = nn.Linear(self.qformer.config.hidden_size, self.embed_dim)
        if self.config.criterion.get('vtm_cat_text_cls', False):
            self.itm_head = nn.Linear(2 * self.qformer.config.hidden_size, 2)
        else:
            self.itm_head = nn.Linear(self.qformer.config.hidden_size, 2)

        # criterions
        self.loss_weight = config.criterion.loss_weight
        self.criterion_vtc_vtm = VTC_VTM_Loss(config.criterion.vtm_hard_neg)

        # init
        trunc_normal_(self.vision_temp_embed)
        trunc_normal_(self.query_tokens)

        self.vision_proj.apply(self._init_weights)
        self.text_proj.apply(self._init_weights)
        self.itm_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image, text, idx):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()

        vision_embeds, vision_query_embeds, vision_past_key_values = self.encode_vision(
            image, return_key_values=True
        )
        text_embeds, pooled_text_embeds = self.encode_text(text)

        # obtain vision and text representations.
        vision_proj = self.vision_proj(vision_query_embeds)  # [B, T, L, C]
        text_proj = self.text_proj(pooled_text_embeds)  # [B, C]

        # calculate loss

        ## VTC loss
        if self.loss_weight.vtc != 0:
            # sim_idx: (sim_v2t, idx), to save computation
            loss_vtc = self.criterion_vtc_vtm.vtc_loss(
                vision_proj,
                text_proj,
                idx,
                self.temp,
                all_gather=True,
                agg_method=self.agg_method,
            )
        else:
            loss_vtc = torch.tensor(0)

        ## VTM loss
        if self.loss_weight.vtm != 0:
            loss_vtm = self.vtm_loss(
                text,
                vision_embeds,
                vision_proj,
                text_proj,
                idx,
            )
        else:
            loss_vtm = torch.tensor(0)

        ## CAP loss
        if self.loss_weight.cap != 0:
            loss_cap = self.cap_loss(
                text,
                vision_past_key_values,
            )
        else:
            loss_cap = torch.tensor(0)

        return dict(
            loss_vtc=loss_vtc * self.loss_weight.vtc,
            loss_vtm=loss_vtm * self.loss_weight.vtm,
            loss_cap=loss_cap * self.loss_weight.cap,
        )

    def freeze_module(self, m):
        m = m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    def encode_vision(self, image, test=False, return_key_values=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].
            - vision_past_key_values (torch.Tensor): The past key values of vision transformer.

        """
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        vision_embeds = self.vision_encoder(image, use_image=use_image)  # [B, T, L, C]
        B, T, L, C = vision_embeds.shape
        vision_embeds = vision_embeds + self.vision_temp_embed.to(vision_embeds.dtype)
        vision_embeds = vision_embeds.reshape(B, -1, C) # [B, T*L, C]
        vision_embeds = self.vision_layernorm(vision_embeds)  # [B, T*L, C]

        vision_atts = torch.ones(
            vision_embeds.shape[:-1], dtype=torch.long, device=vision_embeds.device
        )

        query_tokens = self.query_tokens.expand(image.size(0), -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            use_cache=True,
            return_dict=True,
        )

        vision_query_embeds = query_output.last_hidden_state
        vision_past_key_values = query_output.past_key_values

        if return_key_values:  # This is used in this model cap loss
            return (
                vision_embeds,
                vision_query_embeds,
                vision_past_key_values,
            )
        else:  # This is to match retrieval.py #19
            return vision_embeds, vision_query_embeds

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min_val, max_val)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if "vit" in encoder_name:
            vision_encoder = build_vit(self.config.model)
        else:
            raise ValueError(f"not implemented: {encoder_name}")

        if self.config.model.vit_add_ln:
            vision_layernorm = nn.LayerNorm(self.vision_width, eps=1e-12)
        else:
            vision_layernorm = nn.Identity()

        vision_temp_embed = nn.Parameter(
            torch.zeros(1, self.config.num_frames, 1, self.vision_width)
        )

        return vision_encoder, vision_layernorm, vision_temp_embed

    @torch.no_grad()
    def get_mask(self, sim, idx=None, idx_all=None):
        """get mask for sim matrix."""
        if idx is not None:
            idx = idx.view(-1, 1)
            idx_all = idx_all.view(1, -1) if idx_all is not None else idx.T
            mask = torch.eq(idx, idx_all).to(sim.device)
        else:
            rank = torch.distributed.get_rank()
            mask = torch.zeros_like(sim)
            bs = sim.size(0)
            mask[:, rank * bs : (rank + 1) * bs].fill_diagonal_(1)

        return mask.bool()

    def vtm_loss(
        self,
        text,
        vision_embeds,
        vision_proj,  # [B, L, C]
        text_proj,  # [B, C]
        idx,
    ):
        """vtm loss."""
        with torch.no_grad():
            sim_v2t, sim_t2v = get_sim(
                vision_proj, text_proj, self.temp, agg_method=self.agg_method
            )
            weights_v2t = F.softmax(sim_v2t, dim=1) + 1e-4  # (N, N)
            weights_t2v = F.softmax(sim_t2v, dim=1) + 1e-4

            mask = self.get_mask(sim_v2t, idx=idx).bool()
            weights_v2t.masked_fill_(mask, 0)
            weights_t2v.masked_fill_(mask, 0)
            weights_v2t = torch.nan_to_num_(
                weights_v2t, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
            weights_t2v = torch.nan_to_num_(
                weights_t2v, nan=1e-2, posinf=1e-2, neginf=1e-2
            )

        # select a negative image for each text
        if self.config.criterion.vtm_hard_neg:
            vision_neg_indices = torch.multinomial(weights_t2v, 1).squeeze()
            text_neg_indices = torch.multinomial(weights_v2t, 1).squeeze()
        else:
            vision_neg_indices = self.get_rand_indices(mask, 1).squeeze()
            text_neg_indices = self.get_rand_indices(mask, 1).squeeze()

        vision_embeds_neg = vision_embeds[vision_neg_indices]  # [B, L, C]
        text_ids_neg = text.input_ids[text_neg_indices]  # [B, L]
        text_atts_neg = text.attention_mask[text_neg_indices]  # [B, L]

        # Concat vision pos and neg
        vision_embeds_pos_neg = torch.cat(
            [vision_embeds, vision_embeds_neg, vision_embeds], dim=0
        )  # [3B, L, C]
        vision_atts_pos_neg = torch.ones(
            vision_embeds_pos_neg.size()[:-1],
            dtype=torch.long,
            device=vision_embeds.device,
        )  # [3B, L]

        # Concat text pos and neg
        text_ids_pos_neg = torch.cat(
            [text.input_ids, text.input_ids, text_ids_neg], dim=0
        )
        text_atts_pos_neg = torch.cat(
            [text.attention_mask, text.attention_mask, text_atts_neg], dim=0
        )

        vl_embeddings = self.vtm_embed(
            text_ids=text_ids_pos_neg,
            text_atts=text_atts_pos_neg,
            vision_embeds=vision_embeds_pos_neg,
            vision_atts=vision_atts_pos_neg,
        )
        logits = self.itm_head(vl_embeddings)

        bs = logits.size(0) // 3
        vtm_labels = logits.new_ones(logits.size(0), dtype=torch.long)
        vtm_labels[bs:] = 0
        loss_vtm = F.cross_entropy(logits, vtm_labels)

        return loss_vtm

    def cap_loss(
        self,
        text,
        past_key_values,
    ):
        """caption loss."""
        text_ids = text.input_ids.clone()
        text_ids[:, 0] = self.tokenizer.bos_token_id
        labels = text_ids.masked_fill(text_ids == self.tokenizer.pad_token_id, -100)

        query_atts = torch.ones(
            text_ids.size(0),
            self.query_tokens.size(1),
            dtype=torch.long,
            device=text_ids.device,
        )
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
        cap_output = self.qformer(
            text_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            return_dict=True,
            labels=labels,
        )

        cap_loss = cap_output.loss

        return cap_loss

    def get_text_encoder(self):
        return None

    @torch.jit.ignore
    def no_weight_decay(self):
        """Do not apply weight decay on these parameters"""
        return {
            "query_tokens",
            "temp",
            "vision_temp_embed",
            "vision_encoder.class_embedding",
            "vision_encoder.positional_embedding",
        }

    def vtm_embed(self, text_ids, text_atts, vision_embeds, vision_atts):
        """vtm embedding."""
        query_tokens = self.query_tokens.expand(text_ids.size(0), -1, -1)
        query_atts = torch.ones(
            query_tokens.size()[:-1], dtype=torch.long, device=vision_embeds.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
        )
        if self.config.criterion.get('vtm_add_text_cls', False):
            tmp_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1)].mean(1)
            vl_embeddings = tmp_embeddings + output_itm.last_hidden_state[:, query_tokens.size(1)]
        elif self.config.criterion.get('vtm_cat_text_cls', False):
            tmp_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1)].mean(1)
            vl_embeddings = torch.cat([tmp_embeddings, output_itm.last_hidden_state[:, query_tokens.size(1)]], dim=1)
        else:
            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1)].mean(1)
        return vl_embeddings