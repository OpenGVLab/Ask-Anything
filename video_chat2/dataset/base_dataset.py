import logging
import os
import random
from torch.utils.data import Dataset

from dataset.utils import load_image_from_path
from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

logger = logging.getLogger(__name__)


class ImageVideoBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["image", "video", "text"]
        self.data_root = None
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.video_reader = None
        self.num_tries = None

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index):
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            anno["image"] = os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path):
        if self.media_type == "image":
            return self.load_and_transform_media_data_image(index, data_path)
        else:
            return self.load_and_transform_media_data_video(index, data_path)

    def load_and_transform_media_data_image(self, index, data_path, dynamic_config=None):
        image = load_image_from_path(data_path, client=self.client)
        
        if dynamic_config:
            local_size = dynamic_config["local_size"]
            hd_num = dynamic_config["hd_num"]
            padding = dynamic_config["padding"]
            if padding:
                image = HD_transform_padding(image.float(), image_size=local_size, hd_num=hd_num)
            else:
                image = HD_transform_no_padding(image.float(), image_size=local_size, hd_num=hd_num)

        image = self.transform(image)
        return image, index

    def load_and_transform_media_data_video(self, index, data_path, return_fps=False, clip=None, dynamic_config=None):
        for _ in range(self.num_tries):
            try:
                max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
                frames, frame_indices, fps = self.video_reader(
                    data_path, self.num_frames, self.sample_type, 
                    max_num_frames=max_num_frames, client=self.client, clip=clip
                )
            except Exception as e:
                logger.warning(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement"
                )
                index = random.randint(0, len(self) - 1)
                ann = self.get_anno(index)
                data_path = ann["image"]
                continue

            if dynamic_config:
                local_size = dynamic_config["local_size"]
                hd_num = dynamic_config["hd_num"]
                padding = dynamic_config["padding"]
                if padding:
                    frames = HD_transform_padding(frames.float(), image_size=local_size, hd_num=hd_num)
                else:
                    frames = HD_transform_no_padding(frames.float(), image_size=local_size, hd_num=hd_num)

            # shared aug for video frames
            frames = self.transform(frames)
            if return_fps:
                if fps == None:
                    sec = None
                else:
                    sec = [str(round(f / fps, 1)) for f in frame_indices]
                return frames, index, sec
            else:
                return frames, index
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )
