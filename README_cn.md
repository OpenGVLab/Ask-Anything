# Ask-Anything \[[Paper\]](https://arxiv.org/pdf/2305.06355.pdf)

目前，Ask-Anything是一个简单而有趣的与视频聊天工具。
我们的团队正在努力建立一个智能且强大的用于视频理解的聊天机器人。
<a href="https://huggingface.co/spaces/OpenGVLab/VideoChatGPT"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"> [VideoChat-7B-8Bit] End2End ChatBOT for video and image. </a>
<br>
 <a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/ynhe/AskAnything">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Huggingface">
</a> | <a src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord" href="https://discord.gg/A2Ex6Pph6A">
    <img src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord">
</a> | <a src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud" href="https://ask.opengvlab.com">
    <img src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud"> 
</a>

🚀: 我们通过**指令微调**更新了`video_chat`！相关内容可见我们的[技术报告](https://arxiv.org/pdf/2305.06355.pdf)。相关的**指令微调数据**可见[InternVideo](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data)。`video_chat`之前版本已经移动到`video_chat_with_chatGPT`。

⭐️: 我们还在进行更新版本的开发，敬请期待！

# :movie_camera: 在线演示Demo


<div align=center><h2><a href="https://ask.opengvlab.com">单击此处与ChatVideo一起边看视频边聊天!</a></h2></div>

![f9f8acb1a6da540b60ef8660cf26cac](https://github.com/OpenGVLab/Ask-Anything/assets/43169235/c7e115c6-7eaa-461b-ab07-0118d3b54b11)


# :fire: 更新
- 2023/11/29 VideoChat2和MVBench发布
  - [VideoChat2](./video_chat2/)是基于[UMT](https://github.com/OpenGVLab/unmasked_teacher)和[Vicuna-v0](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)构建的强大基线
  - **1.9M** 多样[指令数据](./video_chat2/data.md)以便有效调优
  - [MVBench](./video_chat2/MVBench.md)是一个全面的视频理解基准

- 2023/05/11 端到端VideoChat
  - [VideoChat](./video_chat/): 基于**指令微调**的视频聊天机器人（也支持图像聊天）
  - [论文](https://arxiv.org/pdf/2305.06355.pdf): 我们展示了如何制作具有两个版本的VideoChat（通过文本和特征），同时还讨论了其背景、应用等方面。

- 2023/04/25 与ChatGPT一起看超过1分钟的视频
  - [VideoChat LongVideo](https://github.com/OpenGVLab/Ask-Anything/tree/long_video_support/): 使用langchain和whisper处理长时信息

- 2023/04/21 与MOSS一起看视频
  - [video_chat_with_MOSS](./video_chat_with_MOSS/): 将视频与MOSS显式编码

- 2023/04/20: 与StableLM一起看视频
  - [VideoChat with StableLM](./video_chat_with_StableLM/): 将视频与StableLM显式编码

- 2023/04/19: 代码发布和在线演示Demo发布
  - [VideoChat with ChatGPT](./video_chat_with_ChatGPT): 将视频与ChatGPT显式编码，对时序信息敏感 [demo is avaliable!](https://ask.opengvlab.com)
  - [MiniGPT-4 for video](./video_miniGPT4/): 将视频与Vicuna隐式编码， 对时序信息不敏感。 ([MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)的简单拓展，将来会改进。)

# 🌤️ 交流群

如果您在试用、运行、部署中有任何问题，欢迎加入我们的微信群讨论！如果您对项目有任何的想法和建议，欢迎加入我们的微信群讨论！

<p align="center"><img width="300" alt="image" src="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/papers/media/wechatv.jpg"></p> 

# :speech_balloon: 示例
https://user-images.githubusercontent.com/24236723/233631602-6a69d83c-83ef-41ed-a494-8e0d0ca7c1c8.mp4

# :page_facing_up: 引用

如果您在研究中发现这个项目对您有帮助，请考虑引用：
```BibTeX
@article{2023videochat,
  title={VideoChat: Chat-Centric Video Understanding},
  author={Li, Kunchang and He, Yinan and Wang, Yi and Li, Yizhuo and Wang, Wenhai and Luo, Ping and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2305.06355},
  year={2023}
}
```

# :hourglass_flowing_sand: 招聘启事

我们的团队不断研究通用视频理解和长期视频推理

我们正在招聘上海人工智能实验室通用视觉组的研究员、工程师和实习生。如果您有兴趣与我们合作，请联系[Yi Wang](https://shepnerd.github.io/) (`wangyi@pjlab.org.cn`).

