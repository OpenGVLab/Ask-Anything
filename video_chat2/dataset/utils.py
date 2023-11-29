from utils.distributed import is_main_process, get_rank, get_world_size
import logging
import torch.distributed as dist
import torch
import io
import os
import json
import re
import numpy as np
from os.path import join
from tqdm import trange
from PIL import Image
from PIL import ImageFile
from torchvision.transforms import PILToTensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def load_image_from_path(image_path, client):
    if image_path.startswith('s3') or image_path.startswith('p2'):
        value = client.Get(image_path)
        img_bytes = np.frombuffer(value, dtype=np.uint8)
        buff = io.BytesIO(img_bytes)
        image = Image.open(buff).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    return image


def load_anno(ann_file_list):
    """[summary]

    Args:
        ann_file_list (List[List[str, str]] or List[str, str]):
            the latter will be automatically converted to the former.
            Each sublist contains [anno_path, image_root], (or [anno_path, video_root, 'video'])
            which specifies the data type, video or image

    Returns:
        List(dict): each dict is {
            image: str or List[str],  # image_path,
            caption: str or List[str]  # caption text string
        }
    """
    if isinstance(ann_file_list[0], str):
        ann_file_list = [ann_file_list]

    ann = []
    for d in ann_file_list:
        data_root = d[1]
        fp = d[0]
        is_video = len(d) == 3 and d[2] == "video"
        cur_ann = json.load(open(fp, "r"))
        iterator = trange(len(cur_ann), desc=f"Loading {fp}") \
            if is_main_process() else range(len(cur_ann))
        for idx in iterator:
            key = "video" if is_video else "image"
            # unified to have the same key for data path
            if isinstance(cur_ann[idx][key], str):
                cur_ann[idx]["image"] = join(data_root, cur_ann[idx][key])
            else:  # list
                cur_ann[idx]["image"] = [join(data_root, e) for e in cur_ann[idx][key]]
        ann += cur_ann
    return ann


def pre_text(text, max_l=None, pre_text=True):
    if pre_text:
        text = re.sub(r"([,.'!?\"()*#:;~])", '', text.lower())
        text = text.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        text = re.sub(r"\s{2,}", ' ', text)
        text = text.rstrip('\n').strip(' ')

        if max_l:  # truncate
            words = text.split(' ')
            if len(words) > max_l:
                text = ' '.join(words[:max_l])
    else:
        pass
    return text


logger = logging.getLogger(__name__)


def collect_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(
            result_dir, '%s_rank%d.json' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(
            result_dir, '%s_rank%d.pth' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    result = None
    if is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(get_world_size()):
            if is_json:
                result_file = os.path.join(
                    result_dir, '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(
                    result_dir, '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)

    return result


def sync_save_result(result, result_dir, filename, is_json=True, is_list=True):
    """gather results from multiple GPUs"""
    if is_json:
        result_file = os.path.join(
            result_dir, "dist_res", '%s_rank%d.json' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        json.dump(result, open(result_file, 'w'))
    else:
        result_file = os.path.join(
            result_dir, "dist_res", '%s_rank%d.pth' % (filename, get_rank()))
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    if is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(get_world_size()):
            if is_json:
                result_file = os.path.join(
                    result_dir, "dist_res", '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(
                    result_dir, "dist_res", '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)
        if is_json:
            json.dump(result, open(final_result_file, 'w'))
        else:
            torch.save(result, final_result_file)

        logger.info('result file saved to %s' % final_result_file)
    dist.barrier()
    return final_result_file, result


def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


