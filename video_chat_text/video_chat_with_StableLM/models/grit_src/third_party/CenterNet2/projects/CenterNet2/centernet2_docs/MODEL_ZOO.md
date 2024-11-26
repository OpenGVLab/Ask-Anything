# MODEL_ZOO

### Common settings and notes

- Multiscale training is used by default in all models. The results are all reported using single-scale testing. 
- We report runtime on our local workstation with a TitanXp GPU and a Titan RTX GPU.
- All models are trained on 8-GPU servers by default. The 1280 models are trained on 24G GPUs. Reducing the batchsize with the linear learning rate rule should be fine.
- All models can be downloaded directly from [Google drive](https://drive.google.com/drive/folders/1eae1cTX8tvIaCeof36sBgxrXEXALYlf-?usp=sharing).


## COCO

### CenterNet

| Model                                     | val mAP | FPS (Titan Xp/ Titan RTX) | links     |
|-------------------------------------------|---------|---------|-----------|
| CenterNet-S4_DLA_8x                       |  42.5   | 50 / 71 |[config](../configs/CenterNet-S4_DLA_8x.yaml)/[model](https://drive.google.com/file/d/1lNBhVHnZAEBRD66MFaHjm5Ij6Z4KYrJq/view?usp=sharing)|
| CenterNet-FPN_R50_1x                      |  40.2   | 20 / 24 |[config](../configs/CenterNet-FPN_R50_1x.yaml)/[model](https://drive.google.com/file/d/1rVG1YTthMXvutC6jr9KoE2DthT5-jhGj/view?usp=sharing)|

#### Note

- `CenterNet-S4_DLA_8x` is a re-implemented version of the original CenterNet (stride 4), with several changes, including
  - Using top-left-right-bottom box encoding and GIoU Loss; adding regression loss to the center 3x3 region.
  - Adding more positive pixels for the heatmap loss whose regression loss is small and is within the center3x3 region.
  - Using more heavy crop augmentation (EfficientDet-style crop ratio 0.1-2), and removing color augmentations.
  - Using standard NMS instead of max pooling.
  - Using RetinaNet-style optimizer (SGD), learning rate rule (0.01 for each batch size 16), and schedule (8x12 epochs).
- `CenterNet-FPN_R50_1x` is a (new) FPN version of CenterNet. It includes the changes above, and assigns objects to FPN levels based on a fixed size range. The model is trained with standard short edge 640-800 multi-scale training with 12 epochs (1x).


### CenterNet2

| Model                                     | val mAP | FPS (Titan Xp/ Titan RTX) | links     |
|-------------------------------------------|---------|---------|-----------|
| CenterNet2-F_R50_1x                       |   41.7  | 22 / 27  |[config](../configs/CenterNet2-F_R50_1x.yaml)/[model](X)|
| CenterNet2_R50_1x                         |  42.9   | 18 / 24 |[config](../configs/CenterNet2_R50_1x.yaml)/[model](https://drive.google.com/file/d/1Osu1J_sskt_1FaGdfJKa4vd2N71TWS9W/view?usp=sharing)|
| CenterNet2_X101-DCN_2x                    |  49.9   | 6 / 8  |[config](../configs/CenterNet2_X101-DCN_2x.yaml)/[model](https://drive.google.com/file/d/1IHgpUHVJWpvMuFUUetgKWsw27pRNN2oK/view?usp=sharing)|
| CenterNet2_DLA-BiFPN-P3_4x                |  43.8   | 40 / 50|[config](../configs/CenterNet2_DLA-BiFPN-P3_4x.yaml)/[model](https://drive.google.com/file/d/12GUNlDW9RmOs40UEMSiiUsk5QK_lpGsE/view?usp=sharing)|
| CenterNet2_DLA-BiFPN-P3_24x               |  45.6   | 40 / 50  |[config](../configs/CenterNet2_DLA-BiFPN-P3_24x.yaml)/[model](https://drive.google.com/file/d/15ZES1ySxubDPzKsHPA7pYg8o_Vwmf-Mb/view?usp=sharing)|
| CenterNet2_R2-101-DCN_896_4x              |  51.2   | 9 / 13 |[config](../configs/CenterNet2_R2-101-DCN_896_4x.yaml)/[model](https://drive.google.com/file/d/1S7_GE8ZDQBWuLEfKHkxzeF3KBsxsbABg/view?usp=sharing)|
| CenterNet2_R2-101-DCN-BiFPN_1280_4x       |  52.9   | 6 / 8 |[config](../configs/CenterNet2_R2-101-DCN-BiFPN_1280_4x.yaml)/[model](https://drive.google.com/file/d/14EBHNMagBCNTQjOXcHoZwLYIi2lFIm7F/view?usp=sharing)|
| CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST |  56.1   | 3 / 5 |[config](../configs/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST.yaml)/[model](https://drive.google.com/file/d/11ww9VlOi_nhpdsU_vBAecSxBU0dR_JzW/view?usp=sharing)|
| CenterNet2_DLA-BiFPN-P5_640_24x_ST        |  49.2   | 33 / 38 |[config](../configs/CenterNet2_DLA-BiFPN-P5_640_24x_ST.yaml)/[model](https://drive.google.com/file/d/1qsHp2HrM1u8WrtBzF5S0oCoLMz-B40wk/view?usp=sharing)|

#### Note

- `CenterNet2-F_R50_1x` uses Faster RCNN as the second stage. All other CenterNet2 models use Cascade RCNN as the second stage.
- `CenterNet2_DLA-BiFPN-P3_4x` follows the same training setting as [realtime-FCOS](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/FCOS-Detection/README.md).
- `CenterNet2_DLA-BiFPN-P3_24x` is trained by repeating the `4x` schedule (starting from learning rate 0.01) 6 times.
- R2 means [Res2Net](https://github.com/Res2Net/Res2Net-detectron2) backbone. To train Res2Net models, you need to download the ImageNet pre-trained weight [here](https://github.com/Res2Net/Res2Net-detectron2) and place it in `output/r2_101.pkl`.
- The last 4 models in the table are trained with the EfficientDet-style resize-and-crop augmentation, instead of the default random resizing short edge in detectron2. We found this trains faster (per-iteration) and gives better performance under a long schedule.
- `_ST` means using [self-training](https://arxiv.org/abs/2006.06882) using pseudo-labels produced by [Scaled-YOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) on COCO unlabeled images, with a hard score threshold 0.5. Our processed pseudo-labels can be downloaded [here](https://drive.google.com/file/d/1LMBjtHhLp6dYf6MjwEQmzCLWQLkmWPpw/view?usp=sharing).
- `CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST` finetunes from `CenterNet2_R2-101-DCN-BiFPN_1280_4x` for an additional `4x` schedule with the self-training data. It is trained under `1280x1280` but tested under `1560x1560`.

## LVIS v1

| Model                                     |  val mAP box | links     |
|-------------------------------------------|--------------|-----------|
| LVIS_CenterNet2_R50_1x                    |  26.5        |[config](../configs/LVIS_CenterNet2_R50_1x.yaml)/[model](https://drive.google.com/file/d/1gT9e-tNw8uzEBaCadQuoOOP2TEYa4kKP/view?usp=sharing)|
| LVIS_CenterNet2_R50_Fed_1x            |  28.3        |[config](../configs/LVIS_CenterNet2_R50_Fed_1x.yaml)/[model](https://drive.google.com/file/d/1a9UjheMCKax0qAKEwPVpq2ZHN6vpqJv8/view?usp=sharing)|

- The models are trained with repeat-factor sampling.
- `LVIS_CenterNet2_R50_Fed_1x` is CenterNet2 with our federated loss. Check our Appendix D of our [paper](https://arxiv.org/abs/2103.07461) or our [technical report at LVIS challenge](https://www.lvisdataset.org/assets/challenge_reports/2020/CenterNet2.pdf) for references.

## Objects365

| Model                                     |  val mAP| links     |
|-------------------------------------------|---------|-----------|
| O365_CenterNet2_R50_1x                    |  22.6   |[config](../configs/O365_CenterNet2_R50_1x.yaml)/[model](https://drive.google.com/file/d/18fG6xGchAlpNp5sx8RAtwadGkS-gdIBU/view?usp=sharing)|

#### Note
- Objects365 dataset can be downloaded [here](https://www.objects365.org/overview.html).
- The model is trained with class-aware sampling.
