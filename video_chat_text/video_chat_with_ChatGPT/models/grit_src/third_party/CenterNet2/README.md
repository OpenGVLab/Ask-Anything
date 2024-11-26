# Probabilistic two-stage detection
Two-stage object detectors that use class-agnostic one-stage detectors as the proposal network.


<p align="center"> <img src='projects/CenterNet2/centernet2_docs/centernet2_teaser.jpg' align="center" height="150px"> </p>

> [**Probabilistic two-stage detection**](http://arxiv.org/abs/2103.07461),            
> Xingyi Zhou, Vladlen Koltun, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 2103.07461](http://arxiv.org/abs/2103.07461))*         

Contact: [zhouxy@cs.utexas.edu](mailto:zhouxy@cs.utexas.edu). Any questions or discussions are welcomed! 

## Abstract

We develop a probabilistic interpretation of two-stage object detection. We show that this probabilistic interpretation motivates a number of common empirical training practices. It also suggests changes to two-stage detection pipelines. Specifically, the first stage should infer proper object-vs-background likelihoods, which should then inform the overall score of the detector. A standard region proposal network (RPN) cannot infer this likelihood sufficiently well, but many one-stage detectors can. We show how to build a probabilistic two-stage detector from any state-of-the-art one-stage detector. The resulting detectors are faster and more accurate than both their one- and two-stage precursors. Our detector achieves 56.4 mAP on COCO test-dev with single-scale testing, outperforming all published results. Using a lightweight backbone, our detector achieves 49.2 mAP on COCO at 33 fps on a Titan Xp.

## Summary

- Two-stage CenterNet: First stage estimates object probabilities, second stage conditionally classifies objects.

- Resulting detector is faster and more accurate than both traditional two-stage detectors (fewer proposals required), and one-stage detectors (lighter first stage head).

- Our best model achieves 56.4 mAP on COCO test-dev.

- This repo also includes a detectron2-based CenterNet implementation with better accuracy (42.5 mAP at 70FPS) and a new FPN version of CenterNet (40.2 mAP with Res50_1x).

## Main results

All models are trained with multi-scale training, and tested with a single scale. The FPS is tested on a Titan RTX GPU.
More models and details can be found in the [MODEL_ZOO](projects/CenterNet2/centernet2_docs/MODEL_ZOO.md).

#### COCO

| Model                                     |  COCO val mAP |  FPS  |
|-------------------------------------------|---------------|-------|
| CenterNet-S4_DLA_8x                       |  42.5         |   71  |
| CenterNet2_R50_1x                         |  42.9         |   24  |
| CenterNet2_X101-DCN_2x                    |  49.9         |    8  |
| CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST |  56.1         |    5  |
| CenterNet2_DLA-BiFPN-P5_24x_ST            |  49.2         |   38  |


#### LVIS 

| Model                     | val mAP box |
| ------------------------- | ----------- |
| CenterNet2_R50_1x         | 26.5        |
| CenterNet2_FedLoss_R50_1x | 28.3        |


#### Objects365

| Model                                     |  val mAP |
|-------------------------------------------|----------|
| CenterNet2_R50_1x                         |  22.6    |

## Installation

Our project is developed on [detectron2](https://github.com/facebookresearch/detectron2). Please follow the official detectron2 [installation](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). All our code is under `projects/CenterNet2/`. In theory, you should be able to copy-paste `projects/CenterNet2/` to the latest detectron2 release or your own detectron2 repo to run our project. There might be API changes in future detectron2 releases that make it incompatible. 

We use the default detectron2 demo script. To run inference on an image folder using our pre-trained model, run

~~~
python projects/CenterNet2/demo/demo.py --config-file projects/CenterNet2/configs/CenterNet2_R50_1x.yaml --input path/to/image/ --opts MODEL.WEIGHTS models/CenterNet2_R50_1x.pth
~~~

## Benchmark evaluation and training

Please check detectron2 [GETTING_STARTED.md](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for running evaluation and training. Our config files are under `projects/CenterNet2/configs` and the pre-trained models are in the [MODEL_ZOO](projects/CenterNet2/centernet2_docs/MODEL_ZOO.md).


## License

Our code under `projects/CenterNet2/` is under [Apache 2.0 license](projects/CenterNet2/LICENSE). `projects/CenterNet2/centernet/modeling/backbone/bifpn_fcos.py` are from [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), which follows the original [non-commercial license](https://github.com/aim-uofa/AdelaiDet/blob/master/LICENSE). The code from detectron2 follows the original [Apache 2.0 license](LICENSE).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2021probablistic,
      title={Probabilistic two-stage detection},
      author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:2103.07461},
      year={2021}
    }
