# MiniGPT-4 for video

Currently, this is a simple extension of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) without extra training. We try to undermine its ability for video understanding with simple prompt design. 


# :fire: Updates
- **2023/04/19**: Simple extension release
    We simple encode 4 frames and use the time-sensitive prompt as follows:
    ```python
        "First, <Img><ImageHere></Img>. Then, <Img><ImageHere></Img>. "
        "After that, <Img><ImageHere></Img>. Finally, <Img><ImageHere></Img>. "
    ```
    However, without video-text instruction finetuning, it's difficult to answer those questions about the time.


# :speech_balloon: Example

![images](./assert/yoga.png)


# :running: Usage
Please follow the instrction in [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4#getting-started) to prepare the environment.
- Prepare the envirment.
    ```shell
        conda env create -f environment.yml
        conda activate minigpt4
    ```
- Download [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) model:
    - ViT: `wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth`
    - QFormer: `wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth`
    - Change the `vit_model_path` and `q_former_model_path` in [minigpt4.yaml](./minigpt4/configs/models/minigpt4.yaml).
- Download [Vicuna](https://github.com/lm-sys/FastChat) model:
    - LLAMA: Download it from the [original repo](https://github.com/facebookresearch/llama) or [hugging face](https://huggingface.co/decapoda-research/llama-13b-hf).
    - If you download LLAMA from the original repo, please process it via the following command:
    ```shell
        # convert_llama_weights_to_hf is copied from transformers
        python src/transformers/models/llama/convert_llama_weights_to_hf.py \
            --input_dir /path/to/downloaded/llama/weights \
            --model_size 7B --output_dir /output/path
    ```
    - Download [Vicuna-13b-deelta-v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0) and process it:
    ```shell
        # fastchat v0.1.10
        python3 -m fastchat.model.apply_delta \
        --base /path/to/llama-13b \
        --target /output/path/to/vicuna-13b \
        --delta lmsys/vicuna-13b-delta-v1.0
    ```
    - Change the `llama_model` in [minigpt4.yaml](./minigpt4/configs/models/minigpt4.yaml).
- Download [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) model:
    - Linear layer can be downloaded [here](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link).
    - Change the `ckpt` in [minigpt4_eval.yaml](./eval_configs/minigpt4_eval.yaml).
- Running demo:
    ```shell
        python demo_video.py --cfg-path eval_configs/minigpt4_eval.yaml
    ```



# Acknowledgement

This project is mainly based on [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), which is support by [Lavis](https://github.com/salesforce/LAVIS), [Vicuna](https://github.com/lm-sys/FastChat) and [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2). Thanks for these amazing projects!