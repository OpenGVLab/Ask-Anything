# 🦜 VideoChat [[paper](https://arxiv.org/abs/2305.06355)/[demo](https://vchat.opengvlab.com/)/[中文文档](README_CN.md)]

![images](assert/framework.png)
In this study, we initiate an exploration into video understanding by introducing VideoChat, an **end-to-end chat-centric video understanding system**. It integrates video foundation models and large language models via a learnable neural interface, excelling in **spatiotemporal reasoning, event localization, and causal relationship inference**. To instructively tune this system, we propose a **video-centric instruction dataset**, composed of thousands of videos matched with detailed descriptions and conversations. This dataset emphasizes **spatiotemporal reasoning and causal relationships**, providing a valuable asset for training chat-centric video understanding systems. Preliminary qualitative experiments reveal our system’s potential across a broad spectrum of video applications and set the standard for future research.


# :fire: Updates
- **2023/11/29** VideoChat2 and MVBench are released.
  - [VideoChat2](../video_chat2/) is a strong baseline built on [UMT](https://github.com/OpenGVLab/unmasked_teacher) and [Vicuna-v0](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md).
  - **1.9M** diverse [instruction data](../video_chat2/DATA.md) are released for effective tuning.
  - [MVBench](../video_chat2/MVBENCH.md) is a comprehensive benchmark for video understanding.

- **2023/06/09**: Release code and scripts for pre-training and instruction tuning:
    - Simply run the [scripts](./scripts) like `bash ./exp/run_7b_stage1.sh`.
    - You can change the `NNODE` and set `MASTER_NODE` by yourself. For stage1, it requires at least 8 GPUs for fast training. For stage2, 4 GPU is enough.
- **2023/05/24**: Release the Stage-pretrained models:
    - [**Model-7B-stage1**](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/video_chat/videochat_7b_stage1.pth), [**Model-13B-stage1**](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/video_chat/videochat_13b_stage1.pth)
- **2023/05/12**: Release the 7B version:
    - 🎊 [**Model-7B**](https://drive.google.com/file/d/1C4s65TC5Zr85I8dZmnfrrw6oDAjj1H4P/view?usp=sharing): 7B requires ~20GB GPU memory, while 13B requires ~32GB GPU memory.
- **2023/05/11**: Release the 🦜**VideoChat V1**, which can **handle both image and video understanding!**
    - 🎊 [**Model-13B**](https://drive.google.com/file/d/1BqmWHWCZBPkhTNWDAq0IfGpbkKLz9C0V/view?usp=share_link) and [**Data**](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data).
    - 🤗 [**Online Demo**](https://vchat.opengvlab.com/)

# :hourglass_flowing_sand: Schedule

- [x] Small-scale video instuction data and tuning
- [x] Instruction tuning on BLIP+UniFormerV2+Vicuna
- [x] Large-scale and complex video instuction data
- [x] Instruction tuning on strong video foundation model
- [ ] User-friendly interactions with longer videos
- [ ] ...

# :speech_balloon: Example [Online🦜](https://vchat.opengvlab.com/)

<div align="center">
<b>
  <font size="4">Comparison with ChatGPT, MiniGPT-4, LLaVA and mPLUG-Owl. </font>
  <br>
  <font size="4" color="red">Our VideoChat can handle both image and video understanding well!</font>
</b>
</div>
<div align="center">
<img src="assert/comparison.png" width="90%">
</div>


<div align="center">
  <font size="4">
	<a href="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/jesse_dance.mp4">[Video]</a> <b>Why the video is funny?</b>
  </font>
</div>
<div align="center">
<img src="assert/humor.png" width="50%">
</div>

<div align="center">
  <font size="4">
	<a href="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/jp_dance.mp4">[Video]</a> <b>Spatial perception</b>
  </font>
</div>
<div align="center">
<img src="assert/spatial.png" width="50%">
</div>

<div align="center">
  <font size="4">
	<a href="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/car_accident.mp4">[Video]</a> <b>Temporal perception</b>
  </font>
</div>
<div align="center">
<img src="assert/temporal.png" width="50%">
</div>

<div align="center">
  <font size="4">
	<a href="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/idol_dancing.mp4">[Video]</a> <b>Multi-turn conversation</b>
  </font>
</div>
<div align="center">
<img src="assert/multi_turn.png" width="50%">
</div>

<div align="center">
  <font size="4">
	<b>Image understanding</b>
  </font>
</div>
<div align="center">
<img src="assert/image.png" width="100%">
</div>

# :running: Usage

- Prepare the envirment.
    ```shell
    pip install -r requirements.txt
    ```
    
- Download [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) model:
    - ViT: `wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth`
    - QFormer: `wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth`
    - Change the `vit_model_path` and `q_former_model_path` in [config.json](./configs/config.json) or [config_7b.json](./configs/config_7b.json).
    
- Download [StableVicuna](https://huggingface.co/CarperAI/stable-vicuna-13b-delta) model:
    - LLAMA: Download it from the [original repo](https://github.com/facebookresearch/llama) or [hugging face](https://huggingface.co/decapoda-research/llama-13b-hf).
    - If you download LLAMA from the original repo, please process it via the following command:
    ```shell
    # convert_llama_weights_to_hf is copied from transformers
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
      --input_dir /path/to/downloaded/llama/weights \
      --model_size 13B --output_dir /output/path
    ```
    - For 13B: Download [stable-vicuna-13b-delta](https://huggingface.co/CarperAI/stable-vicuna-13b-delta) and process it:
      
      You can download `apply_delta.py` from [huggingface](https://huggingface.co/CarperAI/stable-vicuna-13b-delta/raw/main/apply_delta.py)
    ```shell
    # fastchat v0.1.10
    python3 apply_delta.py \
      --base /path/to/model_weights/llama-13b \
      --target stable-vicuna-13b \
      --delta CarperAI/stable-vicuna-13b-delta
    ```
    - For 7B: Download [vicuna-7b-delta-v0](https://huggingface.co/lmsys/vicuna-7b-delta-v0) and process it:
    ```shell
    # fastchat v0.1.10
    python3 apply_delta.py \
      --base /path/to/model_weights/llama-7b \
      --target vicuna-7b-v0 \
      --delta lmsys/vicuna-7b-delta-v0
    ```
    - Change the `llama_model_path` in [config.json](./configs/config.json) or [config_7b.json](./configs/config_7b.json).
    
- Download [VideoChat-13B](https://drive.google.com/file/d/1BqmWHWCZBPkhTNWDAq0IfGpbkKLz9C0V/view?usp=share_link) or [VideoChat-7B](https://drive.google.com/file/d/1C4s65TC5Zr85I8dZmnfrrw6oDAjj1H4P/view?usp=sharing):
    - Change the `videochat_model_path` in [config.json](./configs/config.json)or [config_7b.json](./configs/config_7b.json).
    
- Running demo with Gradio:
    ```shell
    python demo.py
    ```
    
- Another demo on Jupyter Notebook can found in [demo.ipynb](demo.ipynb)


# :robot: Instruction Tuning

- Simply run the scripts.
    ```shell
    bash ./exp/run_7b_stage1.sh
    bash ./exp/run_7b_stage2.sh
    ```
- You can change the `NNODE` and set `MASTER_NODE` by yourself. For stage1, it requires at least 8 GPUs for fast training. For stage2, 4 GPU is enough.


# :page_facing_up: Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{2023videochat,
  title={VideoChat: Chat-Centric Video Understanding},
  author={KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao},
  journal={arXiv preprint arXiv:2305.06355},
  year={2023}
}
```

# :thumbsup: Acknowledgement

Thanks to the open source of the following projects:

[InternVideo](https://github.com/OpenGVLab/InternVideo), [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA), [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2), [StableLM](https://github.com/Stability-AI/StableLM).
