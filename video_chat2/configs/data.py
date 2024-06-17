import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

data_dir = '/mnt/petrelfs/yanziang/datas'
if data_dir is None:
    raise ValueError("please set environment `VL_DATA_DIR` before continue")

data_root = __os.path.join(data_dir, "videos_images")
anno_root_pt = __os.path.join(data_dir, "anno_pretrain")

# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining datasets
    cc3m=[
        f"{data_dir}/cc3m/cc3m_train.json", 
        f"s3://GCC/CC12m",
    ],
    cc12m=[
        f"{data_dir}/cc12m/cc12m_train.json", 
        f"s3://GCC/GCC12m",
    ],
    sbu=[
        f"{data_dir}/sbu.json", 
        f"s3://SBU/images",
    ],
    vg=[
        f"{data_dir}/vg.json", 
        f"s3://VG_dataset/images",
    ],
    coco=[
        f"{data_dir}/coco.json", 
        f"{data_root}/coco",
    ],
    webvid=[
        f"{data_dir}/webvid_10m_train.json", 
        f"s3://WebVid10M",
        "video"
    ],
    webvid_10m=[
        f"{data_dir}/webvid_10m_train.json", 
        f"s3://WebVid10M",
        "video",
    ],
    internvid_10m=[
        f"{anno_root_pt}/internvid_10m_train.json",
        f"{data_root}/internvid_10m",
        "video"
    ],
)

# composed datasets.
available_corpus["msrvtt_1k_test"] = [
    f"{data_dir}/msrvtt_test1k.json",
    f"s3://vicu7/MSRVTT_Videos",
    "video",
]

available_corpus["webvid10m_cc14m"] = [
    available_corpus["webvid_10m"],
    # available_corpus["cc3m"],
    # available_corpus["cc12m"],
    # available_corpus["sbu"]
]
available_corpus["webvid10m_cc14m_plus"] = [
    available_corpus["webvid_10m"],
    # available_corpus["cc3m"],
    # available_corpus["coco"],
    # available_corpus["vg"],
    # available_corpus["sbu"],
    # available_corpus["cc12m"],
    # available_corpus["internvid_10m"],
]