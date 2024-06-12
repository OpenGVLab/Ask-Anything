import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.sampler import StatefulDistributedSampler
from dataset.dataloader import MetaLoader, MetaLoader_rs
from dataset.pt_dataset import PTImgTrainDataset, PTVidTrainDataset, PTImgEvalDataset, PTVidEvalDataset
from dataset.it_dataset import ITImgTrainDataset, ITVidTrainDataset
from dataset.it_dataset_mistral import (
    ITImgTrainDataset_mistral, 
    ITVidTrainDataset_mistral,
    ITTextTrainDataset_mistral,
)
from dataset.it_dataset_phi import ITImgTrainDataset_phi, ITVidTrainDataset_phi

import logging

logger = logging.getLogger(__name__)


def get_media_type(dataset_config):
    if len(dataset_config) >= 3 and dataset_config[2] == "video":
        return "video"
    elif len(dataset_config) >= 3 and dataset_config[2] == "text":
        return "text"
    else:
        return "image"


def create_dataset(dataset_type, config):
    if "clip" in config.model.get("vit_model", 'vit'):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        vision_enc_name = config.model.vision_encoder.name
        if "swin" in vision_enc_name or "vit" in vision_enc_name:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif "beit" in vision_enc_name:
            mean = (0.5, 0.5, 0.5)  # for all beit model except IN1K finetuning
            std = (0.5, 0.5, 0.5)
        elif "clip" in vision_enc_name:
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
        else:
            raise ValueError

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    if config.inputs.video_input.random_aug:
        aug_transform = transforms.RandAugment()
    else:
        aug_transform = transforms.Lambda(lambda x: x)

    if config.model.get('dynamic_config', None):
        logger.info("No training augmentation when finetuning with dynamic resolution.")
        train_transform = transforms.Compose(
            [
                type_transform,
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                aug_transform,
                transforms.RandomResizedCrop(
                    config.inputs.image_res,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                type_transform,
                normalize,
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )

    video_reader_type = config.inputs.video_input.get("video_reader_type", "decord")
    video_only_dataset_kwargs_train = dict(
        video_reader_type=video_reader_type,
        sample_type=config.inputs.video_input.sample_type,
        num_frames=config.inputs.video_input.num_frames,
        num_tries=3,  # false tolerance
    )
    video_only_dataset_kwargs_eval = dict(
        video_reader_type=video_reader_type,
        sample_type=config.inputs.video_input.sample_type_test,
        num_frames=config.inputs.video_input.num_frames_test,
        num_tries=1,  # we want to have predictions for all videos
    )

    if dataset_type == "pt_train":
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file[0], str) else config.train_file
        )
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))

        train_datasets = []
        for m in train_media_types:
            dataset_cls = PTImgTrainDataset if m == "image" else PTVidTrainDataset
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            for train_file in _train_files:
                dataset_kwargs = dict(
                    ann_file=train_file,
                    transform=train_transform,
                    pre_text=config.get(
                        "pre_text", True
                    ),
                )
                if m == "video":
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                datasets.append(dataset_cls(**dataset_kwargs))
            dataset = ConcatDataset(datasets)
            train_datasets.append(dataset)
        return train_datasets
    
    elif dataset_type in ["it_train"]:
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file[0], str) else config.train_file
        )
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))

        train_datasets = []
        for m in train_media_types:
            dataset_cls = ITImgTrainDataset if m == "image" else ITVidTrainDataset
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            for train_file in _train_files:
                dataset_kwargs = dict(
                    ann_file=train_file,
                    transform=train_transform,
                    system=config.model.get("system", ""),
                    start_token=config.model.get("img_start_token", "<Image>"), 
                    end_token=config.model.get("img_end_token", "</Image>"),
                )
                if m == "video":
                    video_only_dataset_kwargs_train.update({
                        "start_token": config.model.get("start_token", "<Video>"),
                        "end_token": config.model.get("end_token", "</Video>"),
                    })
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                    if "tgif" in train_file[1]:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "gif"
                        })
                        dataset_kwargs.update(video_only_dataset_kwargs_train)
                    else:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "decord"
                        })
                        dataset_kwargs.update(video_only_dataset_kwargs_train)
                datasets.append(dataset_cls(**dataset_kwargs))
            dataset = ConcatDataset(datasets)
            train_datasets.append(dataset)
        return train_datasets

    elif dataset_type in ["it_mistral_train"]:
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file[0], str) else config.train_file
        )
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))

        train_datasets = []
        for m in train_media_types:
            if m == "image":
                dataset_cls = ITImgTrainDataset_mistral
            elif m == "video":
                dataset_cls = ITVidTrainDataset_mistral
            elif m == "text":
                dataset_cls = ITTextTrainDataset_mistral
            else:
                raise NotImplementedError
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            for train_file in _train_files:
                dataset_kwargs = dict(
                    ann_file=train_file,
                    transform=train_transform,
                    system=config.model.get("system", ""),
                    start_token=config.model.get("img_start_token", "<Image>"), 
                    end_token=config.model.get("img_end_token", "</Image>"),
                    dynamic_config=config.model.get("dynamic_config", None),
                )
                if m == "video":
                    video_only_dataset_kwargs_train.update({
                        "start_token": config.model.get("start_token", "<Video>"),
                        "end_token": config.model.get("end_token", "</Video>"),
                    })
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                    if "tgif" in train_file[1]:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "gif"
                        })
                        dataset_kwargs.update(video_only_dataset_kwargs_train)
                    else:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "decord"
                        })
                        dataset_kwargs.update(video_only_dataset_kwargs_train)
                datasets.append(dataset_cls(**dataset_kwargs))
            dataset = ConcatDataset(datasets)
            train_datasets.append(dataset)
        return train_datasets
    
    elif dataset_type in ["it_phi_train"]:
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file[0], str) else config.train_file
        )
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))

        train_datasets = []
        for m in train_media_types:
            dataset_cls = ITImgTrainDataset_phi if m == "image" else ITVidTrainDataset_phi
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            for train_file in _train_files:
                dataset_kwargs = dict(
                    ann_file=train_file,
                    transform=train_transform,
                    system=config.model.get("system", ""),
                    start_token=config.model.get("img_start_token", "<Image>"), 
                    end_token=config.model.get("img_end_token", "</Image>"),
                )
                if m == "video":
                    video_only_dataset_kwargs_train.update({
                        "start_token": config.model.get("start_token", "<Video>"),
                        "end_token": config.model.get("end_token", "</Video>"),
                    })
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                    if "tgif" in train_file[1]:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "gif"
                        })
                        dataset_kwargs.update(video_only_dataset_kwargs_train)
                    else:
                        video_only_dataset_kwargs_train.update({
                            "video_reader_type": "decord"
                        })
                        dataset_kwargs.update(video_only_dataset_kwargs_train)
                datasets.append(dataset_cls(**dataset_kwargs))
            dataset = ConcatDataset(datasets)
            train_datasets.append(dataset)
        return train_datasets
    
    elif dataset_type == "pt_eval":
        test_datasets = []
        test_dataset_names = []
        # multiple test datasets, all separate
        for name, data_cfg in config.test_file.items():
            media_type = get_media_type(data_cfg)
            test_dataset_cls = (
                PTImgEvalDataset if media_type == "image" else PTVidEvalDataset
            )
            test_dataset_names.append(name)
            dataset_kwargs = dict(
                ann_file=[data_cfg],
                transform=test_transform,
                has_multi_vision_gt=config.get(
                    "has_multi_vision_gt", False
                ),  # true for ssv2 ret
            )
            if media_type == "video":
                dataset_kwargs.update(video_only_dataset_kwargs_eval)
            test_datasets.append(test_dataset_cls(**dataset_kwargs))
        return test_datasets, test_dataset_names


def create_stateful_sampler(datasets, batch_size):
    samplers = []
    for dataset, bs in zip(datasets, batch_size):
        sampler = StatefulDistributedSampler(dataset, batch_size=bs)
        samplers.append(sampler)
    return samplers


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data
