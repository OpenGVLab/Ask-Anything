from transformers import BertTokenizer
import json
from utils.config import Config
config_file = "configs/config.json"
cfg = Config.from_file(config_file)

import os
import io

from models.videochat2_it import VideoChat2_it
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

print(cfg)
cfg.model.vision_encoder.num_frames = 16
model = VideoChat2_it(config=cfg.model)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    r=16, lora_alpha=32, lora_dropout=0.
)
model.llama_model = get_peft_model(model.llama_model, peft_config)

vc2_model_path = cfg.model.videochat2_model_path
state_dict = torch.load(vc2_model_path, "cpu")

if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False)
else:
    msg = model.load_state_dict(state_dict, strict=False)

model.llama_model = model.llama_model.merge_and_unload()
merged_language_model_path = cfg.merged_language_model_path
model.llama_model.save_pretrained(merged_language_model_path)
model.llama_tokenizer.save_pretrained(merged_language_model_path)