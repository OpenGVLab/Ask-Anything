# Copyright (c) Facebook, Inc. and its affiliates.
import json
import numpy as np
import os
import tempfile
import unittest
import pycocotools.mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_dict, load_coco_json
from detectron2.structures import BoxMode


def make_mask():
    """
    Makes a donut shaped binary mask.
    """
    H = 100
    W = 100
    mask = np.zeros([H, W], dtype=np.uint8)
    for x in range(W):
        for y in range(H):
            d = np.linalg.norm(np.array([W, H]) / 2 - np.array([x, y]))
            if d > 10 and d < 20:
                mask[y, x] = 1
    return mask


def uncompressed_rle(mask):
    l = mask.flatten(order="F").tolist()
    counts = []
    p = False
    cnt = 0
    for i in l:
        if i == p:
            cnt += 1
        else:
            counts.append(cnt)
            p = i
            cnt = 1
    counts.append(cnt)
    return {"counts": counts, "size": [mask.shape[0], mask.shape[1]]}


def make_dataset_dicts(mask, compressed: bool = True):
    """
    Returns a list of dicts that represents a single COCO data point for
    object detection. The single instance given by `mask` is represented by
    RLE, either compressed or uncompressed.
    """
    record = {}
    record["file_name"] = "test"
    record["image_id"] = 0
    record["height"] = mask.shape[0]
    record["width"] = mask.shape[1]

    y, x = np.nonzero(mask)
    if compressed:
        segmentation = mask_util.encode(np.asarray(mask, order="F"))
    else:
        segmentation = uncompressed_rle(mask)
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    obj = {
        "bbox": [min_x, min_y, max_x, max_y],
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": 0,
        "iscrowd": 0,
        "segmentation": segmentation,
    }
    record["annotations"] = [obj]
    return [record]


class TestRLEToJson(unittest.TestCase):
    def test(self):
        # Make a dummy dataset.
        mask = make_mask()
        DatasetCatalog.register("test_dataset", lambda: make_dataset_dicts(mask))
        MetadataCatalog.get("test_dataset").set(thing_classes=["test_label"])

        # Dump to json.
        json_dict = convert_to_coco_dict("test_dataset")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file_name = os.path.join(tmpdir, "test.json")
            with open(json_file_name, "w") as f:
                json.dump(json_dict, f)
            # Load from json.
            dicts = load_coco_json(json_file_name, "")

        # Check the loaded mask matches the original.
        anno = dicts[0]["annotations"][0]
        loaded_mask = mask_util.decode(anno["segmentation"])
        self.assertTrue(np.array_equal(loaded_mask, mask))
        DatasetCatalog.pop("test_dataset")
        MetadataCatalog.pop("test_dataset")

    def test_uncompressed_RLE(self):
        mask = make_mask()
        rle = mask_util.encode(np.asarray(mask, order="F"))
        uncompressed = uncompressed_rle(mask)
        compressed = mask_util.frPyObjects(uncompressed, *rle["size"])
        self.assertEqual(rle, compressed)


class TestConvertCOCO(unittest.TestCase):
    @staticmethod
    def generate_data():
        record = {
            "file_name": "test",
            "image_id": 0,
            "height": 100,
            "width": 100,
            "annotations": [
                {
                    "bbox": [10, 10, 10, 10, 5],
                    "bbox_mode": BoxMode.XYWHA_ABS,
                    "category_id": 0,
                    "iscrowd": 0,
                },
                {
                    "bbox": [15, 15, 3, 3],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                    "iscrowd": 0,
                },
            ],
        }

        return [record]

    def test_convert_to_coco(self):
        DatasetCatalog.register("test_dataset", lambda: TestConvertCOCO.generate_data())
        MetadataCatalog.get("test_dataset").set(thing_classes=["test_label"])
        convert_to_coco_dict("test_dataset")
        DatasetCatalog.pop("test_dataset")
        MetadataCatalog.pop("test_dataset")
