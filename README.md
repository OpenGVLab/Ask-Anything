

# 🦜 VideoChat Family: Ask-Anything 


[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/yinanhe/VideoChat2) | 
<a src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord" href="https://discord.gg/A2Ex6Pph6A">
    <img src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord">
</a> | 
<a src="https://img.shields.io/badge/cs.CV-2305.06355-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2305.06355"> <img src="https://img.shields.io/badge/cs.CV-2305.06355-b31b1b?logo=arxiv&logoColor=red">
</a>| <a src="https://img.shields.io/badge/cs.CV-2311.17005-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2311.17005"> <img src="https://img.shields.io/badge/cs.CV-2311.17005-b31b1b?logo=arxiv&logoColor=red">
</a>| 
<a src="https://img.shields.io/twitter/follow/opengvlab?style=social" href="https://twitter.com/opengvlab">
    <img src="https://img.shields.io/twitter/follow/opengvlab?style=social"> </a>
</a>
<br>
<a href="https://huggingface.co/spaces/OpenGVLab/VideoChatGPT"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"> [VideoChat-7B-8Bit] End2End ChatBOT for video and image. </a> <a href="https://huggingface.co/spaces/OpenGVLab/InternVideo2-Chat-8B-HD"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"> [InternVideo2-Chat-8B-HD]</a>


[中文 README 及 中文交流群](README_cn.md) | [Paper](https://arxiv.org/abs/2305.06355)

<!-- 🚀: We update `video_chat` by **instruction tuning for video & image chatting** now! Find its details [here](https://arxiv.org/pdf/2305.06355.pdf). We release **instruction data** at [InternVideo](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data). The old version of `video_chat` moved to `video_chat_with_chatGPT`.  -->

⭐️: We are also working on a updated version, stay tuned! 
    



# :fire: Updates
- **2025/01/18**: We release [videochat-flash](https://github.com/OpenGVLab/VideoChat-Flash) and [videochat-tpo](https://github.com/OpenGVLab/TPO) to extend MLLMs' capabilities on both long and accurate video understanding. [videochat-flash](https://github.com/OpenGVLab/VideoChat-Flash) sets new records in mutiple video benchmarks (for both short and long videos), improving code usability by leveaging [LLaVA](https://github.com/LLaVA-VL/LLaVA-NeXT) and others. [videochat-tpo](https://github.com/OpenGVLab/TPO) exploits classical vision task annotations (e.g. tracking) to optimize MLLMs in a DPO manner, enhancing MLLMs' performance and enabling capabilities in tracking, segmentation, and more.
- **2024/06/25**: We release the [branch of videochat2 using `vllm`](https://github.com/OpenGVLab/Ask-Anything/tree/vllm), speed up the inference of videochat2.
- **2024/06/19**: 🎉🎉 Our VideoChat2 achieves the best performances among the open-sourced VideoLLMs on [MLVU](https://github.com/JUNJIE99/MLVU), a multi-task long video understanding benchmark.
- **2024/06/13**: Fix some bug and give testing scripts/
    - :warning: We replace some repeated  (~30) QAs in MVBench, which may only affect the results by 0.5%.
    - :loudspeaker: We give the scripts for testing [EgoSchema](https://github.com/egoschema/EgoSchema/tree/main) and [Video-MME](https://github.com/BradyFU/Video-MME/tree/main), please check the [demo_mistral.ipynb](./video_chat2/demo/demo_mistral.ipynb) and [demo_mistral_hd.ipynb](./video_chat2/demo/demo_mistral_hd.ipynb).
- **2024/06/07**: :fire::fire::fire: We release **VideoChat2_HD**, which is fine-tuned with high-resolution data and is capable of handling more diverse tasks. It showcases better performance on different benchmarks, especially for detailed captioning. Furthermore, it achieves **54.8% on [Video-MME](https://github.com/BradyFU/Video-MME/tree/main)**, the best score among 7B MLLMs. Have a try! 🏃🏻‍♀️🏃🏻
- **2024/06/06**: We release **VideoChat2_phi3**, a faster model with robust performaces. 
- **2024/05/22**: We release **VideoChat2_mistral**, which shows better capacity on diverse tasks (**60.4% on MVBench, 78.6% on NExT-QA, 63.8% on STAR, 46.4% on TVQA, 54.4% on EgoSchema-full and 80.5% on IntentQA**). More details have been updated in the paper. 
- 2024/04/05 MVBench is selected as Poster (**Highlight**)!
- 2024/2/27 [MVBench](./video_chat2) is accepted by CVPR2024.
- 2023/11/29 VideoChat2 and MVBench are released.
  - [VideoChat2](./video_chat2/) is a robust baseline built on [UMT](https://github.com/OpenGVLab/unmasked_teacher) and [Vicuna-v0](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md).
  - **2M** diverse [instruction data](./video_chat2/DATA.md) are released for effective tuning.
  - [MVBench](./video_chat2/MVBENCH.md) is a comprehensive benchmark for video understanding.

- 2023/05/11 End-to-end VideoChat and its technical report.
  - [VideoChat1](./video_chat/): Instruction tuning for video chatting (also supports image one).
  - [Paper](https://arxiv.org/pdf/2305.06355.pdf): We present how we craft VideoChat with two versions (via text and embed) along with some discussions on its background, applications, and more.

- 2023/04/25 Watch videos longer than one minute with chatGPT
  - [VideoChat LongVideo](https://github.com/OpenGVLab/Ask-Anything/tree/long_video_support/): Incorporating langchain and whisper into VideoChat.

- 2023/04/21 Chat with MOSS
  - [VideoChat with MOSS](./video_chat_text/video_chat_with_MOSS/): Explicit communication with MOSS. 

- 2023/04/20: Chat with StableLM
  - [VideoChat with StableLM](./video_chat_text/video_chat_with_StableLM/): Explicit communication with StableLM. 

- 2023/04/19: Code release & Online Demo
  - [VideoChat with ChatGPT](./video_chat_with_ChatGPT): Explicit communication with ChatGPT. Sensitive with time. 
  - [MiniGPT-4 for video](./video_chat_text/video_miniGPT4/): Implicit communication with Vicuna. Not sensitive with time. (Simple extension of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), which will be improved in the future.)


<!-- # :speech_balloon: Example
https://user-images.githubusercontent.com/24236723/233631602-6a69d83c-83ef-41ed-a494-8e0d0ca7c1c8.mp4 -->

# 🔨 Getting Started

### Build video chat with:
* [End2End](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat#running-usage)
* [ChatGPT](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_text/video_chat_with_ChatGPT#running-usage)
* [StableLM](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_text/video_chat_with_StableLM#running-usage)
* [MOSS](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_text/video_chat_with_MOSS#running-usage)
* [MiniGPT-4](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_text/video_miniGPT4#running-usage)


# :clapper: [\[End2End ChatBot\]](https://vchat.opengvlab.com)


https://github.com/OpenGVLab/Ask-Anything/assets/24236723/a8667e87-49dd-4fc8-a620-3e408c058e26
    
<video controls>
  <source src="[https://user-images.githubusercontent.com/24236723/233630363-b20304ab-763b-40e5-b526-e2a6b9e9cae2.mp4](https://github.com/OpenGVLab/Ask-Anything/assets/24236723/a8667e87-49dd-4fc8-a620-3e408c058e26)" type="video/mp4">
Your browser does not support the video tag.
</video>


# :movie_camera: [\[Communication with ChatGPT\]](https://vchat.opengvlab.com)

https://user-images.githubusercontent.com/24236723/233630363-b20304ab-763b-40e5-b526-e2a6b9e9cae2.mp4

<video controls>
  <source src="https://user-images.githubusercontent.com/24236723/233630363-b20304ab-763b-40e5-b526-e2a6b9e9cae2.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>


# :page_facing_up: Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{2023videochat,
  title={VideoChat: Chat-Centric Video Understanding},
  author={KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao},
  journal={arXiv preprint arXiv:2305.06355},
  year={2023}
}

@inproceedings{li2024mvbench,
  title={Mvbench: A comprehensive multi-modal video understanding benchmark},
  author={Li, Kunchang and Wang, Yali and He, Yinan and Li, Yizhuo and Wang, Yi and Liu, Yi and Wang, Zun and Xu, Jilan and Chen, Guo and Luo, Ping and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22195--22206},
  year={2024}
}
```

# 🌤️ Discussion Group

If you have any questions during the trial, running or deployment, feel free to join our WeChat group discussion! If you have any ideas or suggestions for the project, you are also welcome to join our WeChat group discussion!


![image](https://github.com/OpenGVLab/Ask-Anything/assets/43169235/9ac44555-7228-415c-be54-6be18df7d79a)

We are hiring researchers, engineers and interns in **General Vision Group, Shanghai AI Lab**.  If you are interested in working with us, please contact [Yi Wang](https://shepnerd.github.io/) (`wangyi@pjlab.org.cn`).
