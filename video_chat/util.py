from decord import VideoReader
from decord import cpu
import numpy as np

import torchvision.transforms as transforms
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)


def loadvideo_decord(sample, sample_rate_scale=1,new_width=384, new_height=384, clip_len=8, frame_sample_rate=2,num_segment=1):
    fname = sample
    vr = VideoReader(fname, width=new_width, height=new_height,
                                     num_threads=1, ctx=cpu(0))
    # handle temporal segments
    converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) //num_segment
    duration = max(len(vr) // vr.get_avg_fps(),8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i*seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer

def loadvideo_decord_origin(sample, sample_rate_scale=1,new_width=384, new_height=384, clip_len=8, frame_sample_rate=2,num_segment=1):
    fname = sample
    vr = VideoReader(fname, 
                                     num_threads=1, ctx=cpu(0))
    # handle temporal segments
    converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) //num_segment
    duration = max(len(vr) // vr.get_avg_fps(),8)

    all_index = []
    for i in range(num_segment):
        index = np.linspace(0, seg_len, num=int(duration))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        index = index + i*seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer


