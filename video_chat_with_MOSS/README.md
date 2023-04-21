# VideoChat with MOSS

VideoChat is a multifunctional video question answering tool that combines the functions of Action Recognition, Visual Captioning and StableLM. Our solution generates dense, descriptive captions for any object and action in a video, offering a range of language styles to suit different user preferences. It supports users to have conversations in different lengths, emotions, authenticity of language.

- Video-Text Generation
- Chat about uploaded video
- Interactive demo

# :fire: Updates

- **2023/04/21**: Code Release

# :speech_balloon: Example

![images](assert/yoga_moss.png)

# :running: Usage

```shell
# Clone the repository:
git clone https://github.com/OpenGVLab/Ask-Anything.git
cd ask-anything/video_chat_with_MOSS

# Install dependencies:
pip install -r requirements.txt

# Download the checkpoints
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth ./pretrained_models/tag2text_swin_14m.pth
wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth ./pretrained_models/grit_b_densecap_objectdet.pth


# Run the VideoChat_with_MOSS gradio demo.
python app.py

```

# Acknowledgement

The project is based on [InternVideo](https://github.com/OpenGVLab/InternVideo), [Tag2Text](https://github.com/xinyu1205/Tag2Text), [GRiT](https://github.com/JialianW/GRiT) and [MOSS](https://github.com/OpenLMLab/MOSS). Thanks for the authors for their efforts.
