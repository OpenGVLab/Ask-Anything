import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

data_dir = 'your_annotation_path'
if data_dir is None:
    raise ValueError("please set environment `VL_DATA_DIR` before continue")

data_root = __os.path.join(data_dir, "videos_images")
anno_root_pt = __os.path.join(data_dir, "anno_pretrain")

# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining datasets
    cc3m=[
        f"{anno_root_pt}/cc3m_train.json", 
        f"{data_root}/cc3m",
    ],
    cc12m=[
        f"{anno_root_pt}/cc12m_train.json", 
        f"{data_root}/cc12m",
    ],
    sbu=[
        f"{anno_root_pt}/sbu.json", 
        f"{data_root}/sbu",
    ],
    vg=[
        f"{anno_root_pt}/vg.json", 
        f"{data_root}/vg",
    ],
    coco=[
        f"{anno_root_pt}/coco.json", 
        f"{data_root}/coco",
    ],
    webvid=[
        f"{anno_root_pt}/webvid_train.json", 
        f"{data_root}/webvid",
        "video"
    ],
    webvid_10m=[
        f"{anno_root_pt}/webvid_10m_train.json", 
        f"{data_root}/webvid_10m",
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
    f"{anno_root_pt}/msrvtt_test1k.json",
    f"{data_root}/MSRVTT_Videos",
    "video",
]

available_corpus["webvid10m_cc14m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_cc14m_plus"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["internvid_10m"],
]