import logging
import os
import json
import sqlite3
import random
from os.path import basename

import numpy as np

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.utils import load_anno, pre_text
from dataset.video_utils import VIDEO_READER_FUNCS
from utils.distributed import is_main_process

logger = logging.getLogger(__name__)


def get_anno_by_id(cur: sqlite3.Cursor, id: int):
    """TODO: Docstring for get_anno_by_id.

    Args:
        cur (sqlite3.Cursor): The dataset cursor.
        id (int): The annotation id.

    Returns:

    """
    pass


class PTImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, pre_text=True):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)

        self.transform = transform
        self.pre_text = pre_text
        logger.info(f"Pre-process text: {pre_text}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        caption = self.anno[index]["caption"]
        anno = {"image": os.path.join(self.data_root, filename), "caption": caption}
        return anno

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data(index, ann["image"])
            caption = pre_text(ann["caption"], pre_text=self.pre_text)
            return image, caption, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class PTVidTrainDataset(PTImgTrainDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3,
        pre_text=True
    ):
        super().__init__(ann_file, transform, pre_text=pre_text)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
