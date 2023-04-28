# Ask-Anything

<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="https://huggingface.co/spaces/ynhe/AskAnything">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Huggingface">
</a> | <a src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord" href="https://discord.gg/A2Ex6Pph6A">
    <img src="https://img.shields.io/discord/1099920215724277770?label=Discord&logo=discord">
</a> | <a src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud" href="https://yinanhe.github.io/projects/chatvideo.html">
    <img src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud"> 
</a> | <a src="https://img.shields.io/twitter/follow/opengvlab?style=social" href="https://twitter.com/opengvlab">
    <img src="https://img.shields.io/twitter/follow/opengvlab?style=social"> 
</a>
<br></br>

[中文 README](README_cn.md)

Currently, Ask-Anything is a simple yet interesting tool for chatting with video.
Our team is trying to build a smart and robust chatbot that can understand video.

We are also working on a updated version, stay tuned! ⭐️





# :movie_camera: Online Demo [\[click here\]](https://yinanhe.github.io/projects/chatvideo.html)

https://user-images.githubusercontent.com/24236723/233630363-b20304ab-763b-40e5-b526-e2a6b9e9cae2.mp4

<video controls>
  <source src="https://user-images.githubusercontent.com/24236723/233630363-b20304ab-763b-40e5-b526-e2a6b9e9cae2.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>


# :fire: Updates
- 2023/04/25 Watch videos longer than one minute with chatGPT
  - [VideoChat_LongVideo](https://github.com/OpenGVLab/Ask-Anything/tree/long_video_support/): Update langchain and whisper to the latest version.

- 2023/04/21 Chat with MOSS
  - [video_chat_with_MOSS](./video_chat_with_MOSS/): Explicit communication with MOSS. 

- 2023/04/20: Chat with StableLM
  - [video_chat_with_StableLM](./video_chat_with_StableLM/): Explicit communication with StableLM. 

- 2023/04/19: Code release & Online Demo
  - [VideoChat](./video_chat/): Explicit communication with ChatGPT. Sensitive with time. [demo is avaliable!](https://yinanhe.github.io/projects/chatvideo.html)
  - [MiniGPT-4 for video](./video_miniGPT4/): Implicit communication with Vicuna. Not sensitive with time. (Simple extension of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), which will be improved in the future.)


# :speech_balloon: Example
https://user-images.githubusercontent.com/24236723/233631602-6a69d83c-83ef-41ed-a494-8e0d0ca7c1c8.mp4

# 🔨 Getting Started

### Build video chat with:
* [ChatGPT](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat#running-usage)
* [StableLM](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_with_StableLM#running-usage)
* [MiniGPT-4](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_miniGPT4#running-usage)
* [MOSS](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat_with_MOSS#running-usage)


# :hourglass_flowing_sand: Ongoing

Our team constantly studies general video understanding and long-term video reasoning:

- [ ] Strong video foundation model.
- [ ] Video-text dataset and video reasoning benchmark.
- [ ] Video-language system with LLMs.
- [ ] Artificial Intelligence Generated Content (AIGC) for Video.
- [ ] ...

We are hiring researchers, engineers and interns in **General Vision Group, Shanghai AI Lab**.  If you are interested in working with us, please contact [Yi Wang](https://shepnerd.github.io/) (`wangyi@pjlab.org.cn`).
