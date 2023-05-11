

# 🦜 [Ask-Anything](https://arxiv.org/pdf/2305.06355.pdf)

<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/ynhe/AskAnything">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Huggingface">
</a> | <a src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord" href="https://discord.gg/A2Ex6Pph6A">
    <img src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord">
</a>   |<a src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat" href="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/wechat_group.jpg"> <img src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat">|<a src="https://img.shields.io/badge/cs.CV-2305.06355-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2305.06355"> <img src="https://img.shields.io/badge/cs.CV-2305.06355-b31b1b?logo=arxiv&logoColor=red">| <a src="https://img.shields.io/twitter/follow/opengvlab?style=social" href="https://twitter.com/opengvlab">
    <img src="https://img.shields.io/twitter/follow/opengvlab?style=social"> </a>
</a>
<br>
<a src="https://img.shields.io/badge/Video%20Chat%20(vChat)-Open-green?logo=alibabacloud" href="https://vchat.opengvlab.com">
    <img src="https://img.shields.io/badge/Video%20Chat%20(vChat)-Open-green?logo=alibabacloud"> End2end ChatBOT for video and image.
<br>    
    <a src="https://img.shields.io/badge/Video%20Chat%20with%20ChatGPT-Open-green?logo=alibabacloud" href="https://ask.opengvlab.com">
    <img src="https://img.shields.io/badge/Video%20Chat%20with%20ChatGPT-Open-green?logo=alibabacloud">  Explicit communication with ChatGPT.  </a>
</a> 


[中文 README 及 中文交流群](README_cn.md) | [Paper](https://arxiv.org/abs/2305.06355)

🚀: We update `video_chat` by **instruction tuning for image & video chatting** now! Old version of `video_chat` moved to `video_chat_with_chatGPT`. We release instruction data at [InternVideo](https://github.com/OpenGVLab/InternVideo/blob/main/Data/instruction_data.md)

⭐️: We are also working on a updated version, stay tuned! 


# :movie_camera: Online Demo [\[click here\]](https://ask.opengvlab.com)

https://user-images.githubusercontent.com/24236723/233630363-b20304ab-763b-40e5-b526-e2a6b9e9cae2.mp4

<video controls>
  <source src="https://user-images.githubusercontent.com/24236723/233630363-b20304ab-763b-40e5-b526-e2a6b9e9cae2.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>


# :fire: Updates
- 2023/05/11 End-to-end VideoChat
  - [VideoChat](./video_chat/): Instruction tuning for image & video chatting.

- 2023/04/25 Watch videos longer than one minute with chatGPT
  - [VideoChat LongVideo](https://github.com/OpenGVLab/Ask-Anything/tree/long_video_support/): Incorporating langchain and whisper into VideoChat.

- 2023/04/21 Chat with MOSS
  - [VideoChat with MOSS](./video_chat_with_MOSS/): Explicit communication with MOSS. 

- 2023/04/20: Chat with StableLM
  - [VideoChat with StableLM](./video_chat_with_StableLM/): Explicit communication with StableLM. 

- 2023/04/19: Code release & Online Demo
  - [VideoChat with ChatGPT](./video_chat_with_ChatGPT): Explicit communication with ChatGPT. Sensitive with time. [demo is avaliable!](https://ask.opengvlab.com)
  - [MiniGPT-4 for video](./video_miniGPT4/): Implicit communication with Vicuna. Not sensitive with time. (Simple extension of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), which will be improved in the future.)


# :speech_balloon: Example
https://user-images.githubusercontent.com/24236723/233631602-6a69d83c-83ef-41ed-a494-8e0d0ca7c1c8.mp4

# 🔨 Getting Started

### Build video chat with:
* [ChatGPT](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat#running-usage)
* [StableLM](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_with_StableLM#running-usage)
* [MiniGPT-4](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_miniGPT4#running-usage)
* [MOSS](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_with_MOSS#running-usage)

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

# :hourglass_flowing_sand: Ongoing

Our team constantly studies general video understanding and long-term video reasoning:

- [ ] Strong video foundation model.
- [ ] Video-text dataset and video reasoning benchmark.
- [ ] Video-language system with LLMs.
- [ ] Artificial Intelligence Generated Content (AIGC) for Video.
- [ ] ...

We are hiring researchers, engineers and interns in **General Vision Group, Shanghai AI Lab**.  If you are interested in working with us, please contact [Yi Wang](https://shepnerd.github.io/) (`wangyi@pjlab.org.cn`).
