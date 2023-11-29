import logging
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from models.utils import allgather_wgrad
from utils.distributed import get_rank, get_world_size
from utils.easydict import EasyDict

logger = logging.getLogger(__name__)


def get_sim(
    vision_proj: torch.Tensor,
    text_proj: torch.Tensor,
    temp=1.0,
    agg_method="mean",
):
    """calculate pair-wise video-text similarity.

    Args:
        vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
        text_proj (torch.Tensor): The text representation. Shape: [B,C].
        temp (torch.Tensor): The temperature. Shape: [].

    Returns: The similarity between video and text. Shape: [B,B].

    """
    vision_proj = F.normalize(vision_proj, dim=-1)
    text_proj = F.normalize(text_proj, dim=-1)
    if vision_proj.ndim == 3:
        sim_v2t = torch.einsum("mld,nd->mln", vision_proj, text_proj) / temp  # [B, L, B]
        sim_t2v = torch.einsum("nd,mld->nlm", text_proj, vision_proj) / temp  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    elif text_proj.ndim == 3:
        sim_v2t = torch.einsum("nd,mld->nlm", vision_proj, text_proj) / temp  # [B, L, B]
        sim_t2v = torch.einsum("nld,md->nlm", text_proj, vision_proj) / temp  # [B, L, B]
        if agg_method == "mean":
            sim_v2t = sim_v2t.mean(1)
            sim_t2v = sim_t2v.mean(1)
        elif agg_method == "max":
            sim_v2t = sim_v2t.max(1)[0]
            sim_t2v = sim_t2v.max(1)[0]
    else:
        sim_v2t = vision_proj @ text_proj.T / temp
        sim_t2v = sim_v2t.T
    return sim_v2t, sim_t2v


class VTC_VTM_Loss(nn.Module):
    """video-text contrastive and matching losses."""

    def __init__(self, vtm_hard_neg):
        super().__init__()
        self.vtm_hard_neg = vtm_hard_neg

    def vtc_loss(
        self,
        vision_proj: torch.Tensor,
        text_proj: torch.Tensor,
        idx: torch.Tensor,
        temp=1.0,
        all_gather=True,
        agg_method="mean",
    ):
        """forward to calculate the loss

        Args:
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            idx (torch.Tensor): The index for each example. Shape: [B,].
            temp (torch.Tensor): The temperature. Shape: [].
            all_gather (bool): If true, will gather samples across all the GPUs and calculate loss across the gathered samples.

        Returns: loss_vtc (torch.Tensor): The video-text contrastive loss. Shape: [].

        """
        if all_gather:
            gather_args = self.get_gather_args()
            vision_proj = allgather_wgrad(vision_proj, gather_args)
            text_proj = allgather_wgrad(text_proj, gather_args)
            if idx is not None:
                idx = allgather_wgrad(idx, gather_args)

        sim_v2t, sim_t2v = get_sim(vision_proj, text_proj, temp, agg_method=agg_method)

        with torch.no_grad():
            sim_v2t_targets = self.get_mask(sim_v2t, idx=idx, normalize=True)
            sim_t2v_targets = sim_v2t_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * sim_v2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2v, dim=1) * sim_t2v_targets, dim=1).mean()

        loss_vtc = (loss_i2t + loss_t2i) / 2
        return loss_vtc

    def vtm_loss(
        self,
        multimodal_encoder,
        vtm_head: nn.Module,
        temp,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        vision_proj: torch.Tensor,
        text_proj: torch.Tensor,
        text_atts: torch.Tensor,
        idx: torch.Tensor,
    ):
        """video-text matching loss.

        Args:
            multinomial_encoder (nn.Module): The multimodal_encoder.
            vtm_head (nn.Module): The head to produce the video-text matching score.
            temp (torch.Tensor): temporature for similarity calculation.
            vision_embeds (torch.Tensor): The features of all patches in the video. Shape: [B,T,L,C].
            text_embeds (torch.Tensor): The features of all tokens in the text. Shape: [B,L,C].
            vision_proj (torch.Tensor): The vision representation. Shape: [B,T,C].
            text_proj (torch.Tensor): The text representation. Shape: [B,C].
            text_atts (torch.Tensor): The padded mask for text tokens. 0 is padded. Shape: [B,L].
            idx (torch.Tensor): The index for each example. Shape: [B,].

        Returns: TODO

        """
        with torch.no_grad():
            sim_v2t, sim_t2v = get_sim(vision_proj, text_proj, temp)
            vision_atts = torch.ones(
                vision_embeds.size()[:-1], dtype=torch.long, device=vision_embeds.device
            )
            weights_v2t = F.softmax(sim_v2t, dim=1) + 1e-4  # (N, N)
            weights_t2v = F.softmax(sim_t2v, dim=1) + 1e-4

            mask = self.get_mask(sim_v2t, idx=idx).bool()
            weights_v2t.masked_fill_(mask, 0)
            weights_t2v.masked_fill_(mask, 0)
            weights_v2t = torch.nan_to_num_(weights_v2t, nan=1e-2, posinf=1e-2, neginf=1e-2)
            weights_t2v = torch.nan_to_num_(weights_t2v, nan=1e-2, posinf=1e-2, neginf=1e-2)

        # select a negative image for each text
        if self.vtm_hard_neg:
            vision_neg_indices = torch.multinomial(weights_t2v, 1).squeeze()
            txt_neg_indices = torch.multinomial(weights_v2t, 1).squeeze()
        else:
            vision_neg_indices = self.get_rand_indices(mask, 1).squeeze()
            txt_neg_indices = self.get_rand_indices(mask, 1).squeeze()

        vision_embeds_neg = vision_embeds[vision_neg_indices]  # [B, T*L, c]
        text_embeds_neg = text_embeds[txt_neg_indices]  # [B, L, d]
        text_atts_neg = text_atts[txt_neg_indices]

        # concat embeddings
        vision_embeds_all = torch.cat([vision_embeds, vision_embeds_neg, vision_embeds], dim=0)
        text_embeds_all = torch.cat([text_embeds, text_embeds, text_embeds_neg], dim=0)
        vision_atts_all = torch.cat([vision_atts, vision_atts, vision_atts], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts, text_atts_neg], dim=0)

        output = multimodal_encoder(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=vision_embeds_all,
            encoder_attention_mask=vision_atts_all,
            return_dict=True,
            mode="fusion",
        )

        vtm_embeds = output.last_hidden_state[:, 0]  # pos (N, d) + neg (2N, d)

        vtm_logits = vtm_head(vtm_embeds)  # [3*B, 2]

        bs = vtm_logits.shape[0] // 3
        vtm_labels = vtm_logits.new_ones(3 * bs, dtype=torch.long)
        vtm_labels[bs:] = 0
        loss_vtm = F.cross_entropy(vtm_logits, vtm_labels)
        return loss_vtm

    def get_rand_indices(self, mask, k):
        """get rand indices according to mask.
        Args:
            mask (torch.Tensor): Shape: (N, L) 0 indicates the positions that we can sample, 1 otherwise
            k (int): the number indices to sample at each row.
        Returns:
            The sampled indices. Shape: [N,k].
            (N, k) indices
        """
        mask = mask.float()
        mask = mask - 10000 * mask
        mask += torch.randn_like(mask)
        _, indices = torch.sort(mask, dim=1, descending=True)
        indices = indices[:, :k].contiguous()
        return indices

    @torch.no_grad()
    def get_mask(self, sim, idx=None, normalize=False):
        """
        Args:
            sim (torch.Tensor): The similarity between videos and texts. shape: (B, B).
            idx (torch.Tensor): The index for each video. Shape: [B].
            normalize (bool): If true, make row sum equal to 1
        """
        if idx is not None:
            idx = idx.view(-1, 1)
            mask = torch.eq(idx, idx.T).to(sim.dtype)
            if normalize:
                mask = mask / mask.sum(1, keepdim=True)
        else:
            mask = torch.zeros_like(sim)
            mask.fill_diagonal_(1)
        return mask  # `1` mark valid/matched location

    @lru_cache(maxsize=16)
    def get_gather_args(self):
        """obtain the args for all_gather
        Returns: dict.

        """
        return EasyDict({"world_size": get_world_size(), "rank": get_rank()})


class MLMLoss(nn.Module):
    """masked language modeling loss."""

    def __init__(self, masking_prob, tokenizer):
        super(MLMLoss, self).__init__()
        self.tokenizer = tokenizer
        self.masking_prob = masking_prob

    def mlm_loss(
        self,
        text_encoder,
        text,
        vision_embeds,
        vision_atts,
    ):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.masking_prob)
        input_ids, labels = self.mask(
            input_ids,
            text_encoder.config.vocab_size,
            input_ids.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )

        intermediate_mlm_output = text_encoder.bert(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            mode="text",
        )

        text_embeds = intermediate_mlm_output.last_hidden_state

        mlm_output = text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            labels=labels,
            soft_labels=None,
            mode="fusion",
        )
        return mlm_output.loss

    def simple_mlm_loss(
        self,
        text_encoder,
        text,
        text_embeds,
        vision_embeds,
        vision_atts,
        labels
    ):
        mlm_output = text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_atts,
            return_dict=True,
            labels=labels,
            soft_labels=None,
            mode="fusion",
        )
        return mlm_output.loss

    def mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            # We only compute loss on masked tokens
            targets[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


class MAC_Loss(nn.Module):
    """mask align clip loss."""

    def __init__(self, mac_norm_type='l2', mac_loss_type='l2'):
        super().__init__()
        self.norm_type = mac_norm_type
        self.loss_type = mac_loss_type
        logger.info(f'Norm type: {mac_norm_type}')
        logger.info(f'Loss type: {mac_loss_type}')

        if mac_loss_type == 'mse':
            self.loss_func = nn.MSELoss()
        elif mac_loss_type == 'smooth_l1':
            self.loss_func = nn.SmoothL1Loss()
    
    def mac_loss(self, student_output, clip_output):
        """forward to calculate the loss

        Args:
            student_output (torch.Tensor): The student output. Shape: [K,B,N,C].
            clip_output (torch.Tensor): The teacher representation. Shape: [K,B,N,C].

        Returns: loss_mac (torch.Tensor): The mask clip alignment loss. Shape: [].
        """

        if self.norm_type == 'l2':
            student_output = student_output / student_output.norm(dim=-1, keepdim=True)
            clip_output = clip_output / clip_output.norm(dim=-1, keepdim=True)
        elif self.norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        if self.loss_type == 'l2':
            loss_mac = (2 - 2 * (student_output * clip_output).sum(dim=-1)).mean()
        elif self.loss_type in ['mse', 'smooth_l1']:
            loss_mac = self.loss_func(input=student_output, target=clip_output)
        else:
            raise NotImplementedError

        return loss_mac

    def mac_vision_loss(self, student_v_output, clip_v_output):
        """forward to calculate the loss

        Args:
            student_v_output (torch.Tensor): The student output. Shape: [B,T,C].
            clip_v_output (torch.Tensor): The teacher representation. Shape: [B,T,C].

        Returns: loss_mac (torch.Tensor): The mask clip alignment loss. Shape: [].
        """

        if student_v_output.shape[1] != clip_v_output.shape[1]:
            student_v_output = student_v_output.mean(1, keepdim=True)
            clip_v_output = clip_v_output.mean(1, keepdim=True)
        if self.norm_type == 'l2':
            student_v_output = student_v_output / student_v_output.norm(dim=-1, keepdim=True)
            clip_v_output = clip_v_output / clip_v_output.norm(dim=-1, keepdim=True)
        elif self.norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        if self.loss_type == 'l2':
            loss_mac = (2 - 2 * (student_v_output * clip_v_output).sum(dim=-1)).mean()
        elif self.loss_type in ['mse', 'smooth_l1']:
            loss_mac = self.loss_func(input=student_v_output, target=clip_v_output)
        else:
            raise NotImplementedError

        return loss_mac

    def mac_all_loss(
        self, 
        student_v_output, clip_v_output,
        student_t_output, clip_t_output,
    ):
        """forward to calculate the loss

        Args:
            student_v_output (torch.Tensor): The student output. Shape: [B,T,C].
            clip_v_output (torch.Tensor): The teacher representation. Shape: [B,T,C].
            student_t_output (torch.Tensor): The student output. Shape: [B,1,C].
            clip_t_output (torch.Tensor): The teacher representation. Shape: [B,1,C].

        Returns: loss_mac (torch.Tensor): The mask clip alignment loss. Shape: [].
        """

        if student_v_output.shape[1] != clip_v_output.shape[1]:
            student_v_output = student_v_output.mean(1, keepdim=True)
            clip_v_output = clip_v_output.mean(1, keepdim=True)
        if self.norm_type == 'l2':
            student_v_output = student_v_output / student_v_output.norm(dim=-1, keepdim=True)
            clip_v_output = clip_v_output / clip_v_output.norm(dim=-1, keepdim=True)
            student_t_output = student_t_output / student_t_output.norm(dim=-1, keepdim=True)
            clip_t_output = clip_t_output / clip_t_output.norm(dim=-1, keepdim=True)
        elif self.norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        if self.loss_type == 'l2':
            loss_mac_v = (2 - 2 * (student_v_output * clip_v_output).sum(dim=-1)).mean()
            loss_mac_t = (2 - 2 * (student_t_output * clip_t_output).sum(dim=-1)).mean()
        elif self.loss_type in ['mse', 'smooth_l1']:
            loss_mac_v = self.loss_func(input=student_v_output, target=clip_v_output)
            loss_mac_t = self.loss_func(input=student_t_output, target=clip_t_output)
        else:
            raise NotImplementedError

        return (loss_mac_v + loss_mac_t) / 2.