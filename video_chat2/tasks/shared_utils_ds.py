import copy
import logging
import os
import os.path as osp
from os.path import join

import torch
import deepspeed
from torch.utils.data import ConcatDataset, DataLoader

from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, find_unused_parameters=False, num_steps_per_epoch=-1,
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    model = model_cls(config=config.model)

    model = model.to(torch.device(config.device))
    if config.fp16:
        if config.get('bf16', True):
            logger.info("Change to bfloat16 for model")
            model = model.to(torch.bfloat16)
        else:
            logger.info("Change to float16 for model")
            model = model.half()
    model_without_ddp = model

    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        optimizer_params = create_optimizer(config.optimizer, model, return_group=True)
        scheduler = None
        scaler = None
    else:
        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.gpu],
                find_unused_parameters=find_unused_parameters,  # `False` for image-only task
            )

        optimizer = create_optimizer(config.optimizer, model)
        scheduler = create_scheduler(config.scheduler, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    start_epoch = 0
    global_step = 0

    # auto resume the latest checkpoint
    if config.get("auto_resume", False):
        logger.info("Auto resuming")
        model_latest = join(config.output_dir, "ckpt_latest.pth")
        model_best = join(config.output_dir, "ckpt_best.pth")

        large_step_num = -1
        large_num = -1
        for p in os.listdir(config.output_dir):
            if 'ckpt_iter' in p:
                num = p.split('_iter')[1].split('.')[0]
                if str.isnumeric(num):
                    if int(num) > large_step_num:
                        large_step_num = int(num)
            elif 'ckpt_' in p:
                num = p.split('_')[1].split('.')[0]
                if str.isnumeric(num):
                    if int(num) > large_num:
                        large_num = int(num)
        if large_step_num != -1:
            logger.info(f"Load the latest step: {large_step_num}")
            model_latest = join(config.output_dir, f"ckpt_iter{large_step_num:02d}.pth")
        if large_num != -1 and (large_num + 1) * num_steps_per_epoch > large_step_num:
            logger.info(f"Load the latest epoch: {large_num}")
            model_latest = join(config.output_dir, f"ckpt_{large_num:02d}.pth")

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            if osp.isdir(model_latest):
                config.pretrained_path = model_latest
                config.resume = True
            elif osp.isdir(model_best):
                config.pretrained_path = model_best
                config.resume = True
            else:
                logger.info(f"Not found checkpoint in {config.output_dir}")
        else:
            if osp.isfile(model_latest):
                config.pretrained_path = model_latest
                config.resume = True
            elif osp.isfile(model_best):
                config.pretrained_path = model_best
                config.resume = True
            else:
                logger.info(f"Not found checkpoint in {config.output_dir}")

    # load pretrained model
    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        logger.info('Use deepspeed to initialize model!!!')
        model = model_without_ddp
        model, optimizer, _, _ = deepspeed.initialize(
            args=config, model=model, model_parameters=optimizer_params, dist_init_required=not config.distributed,
            lr_scheduler=lambda opt: create_scheduler(config.scheduler, opt)
        )
        if osp.isdir(config.pretrained_path):
            logger.info(f"Load pretrained model from {config.pretrained_path}")
            output_dir, tag = os.path.split(config.pretrained_path)
            if config.resume:
                _, client_state = model.load_checkpoint(
                    output_dir, tag=tag, load_module_strict=False,
                    # Resume and decrease the learning rate
                    load_optimizer_states=False, 
                    load_lr_scheduler_states=False, 
                )
                global_step = model.global_steps
                assert num_steps_per_epoch > 0, "Please provide num_steps_per_epoch"
                start_epoch = global_step // num_steps_per_epoch
            else:
                _, client_state = model.load_checkpoint(
                    output_dir, tag=tag, load_module_strict=False, 
                    load_optimizer_states=False, load_lr_scheduler_states=False,
                    load_module_only=True
                )
    else:
        if osp.isfile(config.pretrained_path):
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            logger.info(f"Load pretrained model from {config.pretrained_path}")
            if 'model' in checkpoint.keys():
                state_dict = checkpoint["model"]
            elif 'module' in checkpoint.keys():
                state_dict = checkpoint["module"]
            else:
                state_dict = checkpoint
            # resume optimizer
            if config.resume:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                scaler.load_state_dict(checkpoint["scaler"])
                global_step = checkpoint["global_step"]
                start_epoch = global_step // num_steps_per_epoch

            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info(f"Loaded checkpoint from {config.pretrained_path}")
        else:
            logger.warning("No pretrained checkpoint provided, training from scratch")

    logger.info(f"Cuda memory after create model: {torch.cuda.memory_allocated() // 1024**2}M, Max mem: {torch.cuda.max_memory_allocated() // 1024**2}M")

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    )
