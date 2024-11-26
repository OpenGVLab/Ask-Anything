import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'models/grit_src/third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from models.grit_src.grit.config import add_grit_config

from models.grit_src.grit.predictor import VisualizationDemo
import json


# constants
WINDOW_NAME = "GRiT"


def dense_pred_to_caption(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ""
    for i in range(len(object_description)):
        new_caption += (object_description[i] + ": " + str([int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
    return new_caption

def dense_pred_to_caption_only_name(predictions):
    # boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ",".join(object_description)
    del predictions
    # for i in range(len(object_description)):
    #     new_caption += (object_description[i] + ": " + str([int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
    return new_caption

def setup_cfg(args):
    cfg = get_cfg()
    if args["cpu"]:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    if args["test_task"]:
        cfg.MODEL.TEST_TASK = args["test_task"]
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser(device):
    arg_dict = {'config_file': "models/grit_src/configs/GRiT_B_DenseCap_ObjectDet.yaml", 'cpu': False, 'confidence_threshold': 0.5, 'test_task': 'DenseCap', 'opts': ["MODEL.WEIGHTS", "pretrained_models/grit_b_densecap_objectdet.pth"]}
    if device == "cpu":
        arg_dict["cpu"] = True
    return arg_dict

def image_caption_api(image_src, device):
    args2 = get_parser(device)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image_src:
        img = read_image(image_src, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        new_caption = dense_pred_to_caption(predictions)
    return new_caption

def init_demo(device):
    args2 = get_parser(device)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    return demo

if __name__=="__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='7'
    print(image_caption_api("images/dancing_example_4.mp4_20230417_135359.263.jpg",'cuda'))