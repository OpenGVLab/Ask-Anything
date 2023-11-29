import datetime
import logging
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from models.criterions import get_sim
from utils.basic_utils import MetricLogger
from utils.distributed import get_rank, get_world_size

logger = logging.getLogger(__name__)


def extract_text_feats(texts, max_txt_l, tokenizer, model, device):
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_atts = []

    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_txt_l,
            return_tensors="pt",
        ).to(device)

        text_feat = model.encode_text(text_input)[0]
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    return text_feats, text_atts


def extract_vision_feats(data_loader, model, device, config):
    image_feats_all = []
    pooled_image_feats_all = []
    metric_logger = MetricLogger(delimiter="  ")
    header = "extracting image feats"
    iterator = metric_logger.log_every(data_loader, 100, header)
    for image, img_id in iterator:
        image = image.to(device, non_blocking=True)
        image_feat, pooled_image_feat = model.encode_vision(image, test=True)
        if config.evaluation.eval_frame_ensemble == "concat":  # default
            if len(image_feat.shape) == 4:
                image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
            image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
        else:
            assert config.video_input.num_frames == 1, "only support single-frame"
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
        if config.evaluation.eval_offload:
            image_feats_all.append(image_feat.cpu())
            pooled_image_feats_all.append(pooled_image_feat.cpu())
        else:
            image_feats_all.append(image_feat)
            pooled_image_feats_all.append(pooled_image_feat)

    image_feats_all = torch.cat(image_feats_all, dim=0)

    pooled_image_feats_all = torch.cat(pooled_image_feats_all, dim=0)
    return image_feats_all, pooled_image_feats_all


@torch.no_grad()
def evaluation_wrapper(model, data_loader, tokenizer, device, config, prefix=""):
    with torch.cuda.amp.autocast(enabled=config.fp16):
        i2t_x, t2i_x, i2t_emb, t2i_emb = evaluation(
            model, data_loader, tokenizer, device, config
        )
    score_pairs = [
        (prefix + "/", i2t_x, t2i_x),
        (prefix + "_emb/", i2t_emb, t2i_emb),
    ]
    res = dict()
    for name, i2t, t2i in score_pairs:
        if i2t is not None:
            txt2img_ids = data_loader.dataset.txt2img
            img2txt_ids = data_loader.dataset.img2txt
            res[name] = itm_eval(i2t, t2i, txt2img_ids, img2txt_ids)
    return res


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"
    dtype = torch.half if config.fp16 else torch.float
    media_type = data_loader.dataset.media_type
    logger.info(f"Start evaluation for media_type={media_type}")

    logger.info("Computing dual encoder features...")
    start_time = time.time()

    # this computes all features in each GPU
    texts = data_loader.dataset.text
    max_txt_l = config.inputs.max_txt_l
    if not isinstance(max_txt_l, int):
        max_txt_l = max_txt_l[media_type]
    text_feats, text_atts = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device
    )  # (bsz, Lt, d), (bsz, Lt)

    image_feats, pooled_image_feats = extract_vision_feats(
        data_loader, model, device, config
    )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
    logger.info("Finished feature extraction")
    logger.info("Computing ITC scores [dot-product]")
    _pooled_image_feats = (
        pooled_image_feats.to(device, non_blocking=True)
        if config.evaluation.eval_offload
        else pooled_image_feats
    )
    i2t_scores, t2i_scores = get_sim(
        model.vision_proj(_pooled_image_feats), model.text_proj(text_feats[:, 0])
    )
    logger.info("Computing ITC scores [dot-product], done!")

    num_images = len(data_loader.dataset.image)
    i2t_scores_x = torch.full((num_images, len(texts)), -100.0).to(
        device, torch.float, non_blocking=True
    )

    # computes only part of the scores at each GPU, gather at the end
    logger.info("Rerank dual-encoder results with cross-encoder...")
    num_tasks = get_world_size()
    rank = get_rank()
    # only uses the part associated with the raw eval set
    # compute image2text #
    step = num_images // num_tasks + 1
    start = rank * step
    end = min(num_images, start + step)

    text_encoder = model.get_text_encoder()
    iterator = metric_logger.log_every(i2t_scores[start:end], 100, header)
    logger.info(f"i2t_scores.shape {i2t_scores[start:end].shape}")

    # generate score for each clip, and aggregate all clip scores for a video
    n_clip_per_video = (
        image_feats.shape[1] if not config.deep_fusion else image_feats[0].shape[1]
    )

    logger.info(
        f"n_clip_per_video={n_clip_per_video}, with eval_frame_ensemble={config.evaluation.eval_frame_ensemble}"
    )
    for i, sims in enumerate(iterator):
        k = min(len(sims), config.evaluation.k_test)
        topk_sim, topk_idx = sims.topk(k=k, dim=0)

        clip_scores = []
        for clip_idx in range(n_clip_per_video):
            if config.deep_fusion:
                encoder_output = [
                    feat[start + i, clip_idx].to(device, non_blocking=True)
                    for feat in image_feats
                ]

            else:
                encoder_output = (
                    image_feats[start + i, clip_idx].to(device, non_blocking=True)
                    if config.evaluation.eval_offload
                    else image_feats[start + i, clip_idx]
                )  # (#frm*Li, d)

            """ original
            encoder_output = encoder_output.repeat(k, 1, 1)   # (k=128, #frm*Li, d)
            encoder_att = torch.ones(
                encoder_output.size()[:-1], dtype=torch.long
            ).to(device, non_blocking=True)
            output = text_encoder(
                encoder_embeds=text_feats[topk_idx],
                attention_mask=text_atts[topk_idx],
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode="fusion"
            )

            itm_embeds = output.last_hidden_state[:, 0]
            """

            # new
            bs = 32
            # bs = config.batch_size_test.video
            itm_embeds = []

            if config.deep_fusion:
                if len(topk_idx) % bs != 0:
                    left = len(topk_idx) % bs
                    left_encoder_output = [feat.repeat(left, 1, 1) for feat in encoder_output]
                    left_encoder_att = [
                        torch.ones(feat.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )
                        for feat in left_encoder_output
                    ]
                encoder_output = [feat.repeat(bs, 1, 1) for feat in encoder_output]
                encoder_att = [
                    torch.ones(feat.size()[:-1], dtype=torch.long).to(
                        device, non_blocking=True
                    )
                    for feat in encoder_output
                ]
            else:
                if len(topk_idx) % bs != 0:
                    left = len(topk_idx) % bs
                    left_encoder_output = encoder_output.repeat(left, 1, 1)  # (k=128, #frm*Li, d)
                    left_encoder_att = torch.ones(left_encoder_output.size()[:-1], dtype=torch.long).to(
                        device, non_blocking=True
                    )
                encoder_output = encoder_output.repeat(bs, 1, 1)  # (k=128, #frm*Li, d)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                    device, non_blocking=True
                )

            for j in range(0, len(topk_idx), bs):
                if j + bs > len(topk_idx):
                    output = text_encoder(
                        encoder_embeds=text_feats[topk_idx[j:]],
                        attention_mask=text_atts[topk_idx[j:]],
                        encoder_hidden_states=left_encoder_output,
                        encoder_attention_mask=left_encoder_att,
                        return_dict=True,
                        mode="fusion",
                    )
                else:
                    output = text_encoder(
                        encoder_embeds=text_feats[topk_idx[j : j + bs]],
                        attention_mask=text_atts[topk_idx[j : j + bs]],
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_att,
                        return_dict=True,
                        mode="fusion",
                    )
                batch_itm_embeds = output.last_hidden_state[:, 0]
                itm_embeds.append(batch_itm_embeds)
            itm_embeds = torch.cat(itm_embeds, dim=0)
            # end new

            score = model.itm_head(itm_embeds)[:, 1]
            clip_scores.append(score)

        if len(clip_scores) == 1:
            score = clip_scores[0]
        else:
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
            clip_scores = torch.stack(clip_scores)  # (#clips, k)
            if config.evaluation.eval_frame_ensemble == "mean":
                score = clip_scores.mean(0)
            elif config.evaluation.eval_frame_ensemble == "max":
                score = clip_scores.max(0)[0]
            elif config.evaluation.eval_frame_ensemble == "lse":  # LogSumExp
                score = torch.logsumexp(clip_scores, dim=0)
            else:
                raise ValueError(
                    "config.evaluation.eval_frame_ensemble must in [mean, max, lse] when #clip > 1."
                )

        i2t_scores_x[start + i, topk_idx] = score.to(i2t_scores_x.dtype)

    # compute text2image #
    num_text = len(data_loader.dataset.text)
    t2i_scores_x = torch.full((num_text, len(data_loader.dataset.image)), -100.0).to(
        device, torch.float, non_blocking=True
    )

    step = num_text // num_tasks + 1
    start = rank * step
    end = min(num_text, start + step)

    iterator = metric_logger.log_every(t2i_scores[start:end], 100, header)
    logger.info(f"t2i_scores.shape {t2i_scores[start:end].shape}")
    # generate score for each clip, and aggregate all clip scores for a video
    n_clip_per_video = (
        image_feats.shape[1] if not config.deep_fusion else image_feats[0].shape[1]
    )
    for i, sims in enumerate(iterator):
        k = min(len(sims), config.evaluation.k_test)
        topk_sim, topk_idx = sims.topk(k=k, dim=0)

        clip_scores = []
        for clip_idx in range(n_clip_per_video):

            """old
            encoder_output = image_feats[topk_idx, clip_idx].to(device, non_blocking=True) \
                if config.evaluation.eval_offload else image_feats[topk_idx, clip_idx]
            encoder_att = torch.ones(
                encoder_output.size()[:-1], dtype=torch.long
            ).to(device, non_blocking=True)
            output = text_encoder(
                encoder_embeds=text_feats[start+i].repeat(k, 1, 1),
                attention_mask=text_atts[start+i].repeat(k, 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode="fusion"
            )

            itm_embeds = output.last_hidden_state[:, 0]
            """

            # new
            bs = 32
            # bs = config.batch_size_test.video
            itm_embeds = []
            for j in range(0, len(topk_idx), bs):

                if config.deep_fusion:
                    encoder_output = [
                        feat[topk_idx[j : j + bs], clip_idx].to(device, non_blocking=True)
                        for feat in image_feats
                    ]
                    encoder_att = [
                        torch.ones(feat.size()[:-1], dtype=torch.long).to(
                            device, non_blocking=True
                        )
                        for feat in encoder_output
                    ]
                else:
                    encoder_output = (
                        image_feats[topk_idx[j : j + bs], clip_idx].to(
                            device, non_blocking=True
                        )
                        if config.evaluation.eval_offload
                        else image_feats[topk_idx[j : j + bs], clip_idx]
                    )
                    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                        device, non_blocking=True
                    )

                repeat_n = (
                    encoder_output.shape[0]
                    if not config.deep_fusion
                    else encoder_output[0].shape[0]
                )
                output = text_encoder(
                    encoder_embeds=text_feats[start + i].repeat(repeat_n, 1, 1),
                    attention_mask=text_atts[start + i].repeat(repeat_n, 1),
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_att,
                    return_dict=True,
                    mode="fusion",
                )

                batch_itm_embeds = output.last_hidden_state[:, 0]
                itm_embeds.append(batch_itm_embeds)

            itm_embeds = torch.cat(itm_embeds, dim=0)
            # end new

            score = model.itm_head(itm_embeds)[:, 1]
            clip_scores.append(score)

        if len(clip_scores) == 1:
            score = clip_scores[0]
        else:
            assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
            clip_scores = torch.stack(clip_scores)  # (#clips, k)
            if config.evaluation.eval_frame_ensemble == "mean":
                score = clip_scores.mean(0)
            elif config.evaluation.eval_frame_ensemble == "max":
                score = clip_scores.max(0)[0]
            elif config.evaluation.eval_frame_ensemble == "lse":  # LogSumExp
                score = torch.logsumexp(clip_scores, dim=0)
            else:
                raise ValueError(
                    "config.evaluation.eval_frame_ensemble must in [mean, max, lse] when #clip > 1."
                )

        t2i_scores_x[start + i, topk_idx] = score.to(t2i_scores_x.dtype)

    if config.distributed:
        # gether across GPUs
        dist.barrier()
        dist.all_reduce(i2t_scores_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(t2i_scores_x, op=dist.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Evaluation time {total_time_str}")

    return (
        i2t_scores_x.cpu().numpy(),
        t2i_scores_x.cpu().numpy(),
        i2t_scores.cpu().numpy(),
        i2t_scores.T.cpu().numpy(),
    )


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        gt_txt_ids = img2txt[index]
        if isinstance(gt_txt_ids, int):
            ranks[index] = np.where(inds == gt_txt_ids)[0][0]
        else:
            rank = 1e20
            for i in gt_txt_ids:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        gt_img_ids = txt2img[index]
        if isinstance(gt_img_ids, int):
            ranks[index] = np.where(inds == gt_img_ids)[0][0]
        else:  # list, used in the case each caption has multiple GT images
            # Score
            rank = 1e20
            for i in gt_img_ids:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    }
    eval_result = {k: round(v, 2) for k, v in eval_result.items()}
    return eval_result
