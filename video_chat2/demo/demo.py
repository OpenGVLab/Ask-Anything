import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from conversation import Chat

# videochat
from utils.config import Config
from utils.easydict import EasyDict
# from models.videochat2_it import VideoChat2_it
from models import VideoChat2_it_vicuna as VideoChat2_it
from peft import get_peft_model, LoraConfig, TaskType


# ========================================
#             Model Initialization
# ========================================
def init_model():
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = 4
    # cfg.model.videochat2_model_path = ""
    # cfg.model.debug = True
    model = VideoChat2_it(config=cfg.model)
    model = model.to(torch.device(cfg.device))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.
    )
    model.llama_model = get_peft_model(model.llama_model, peft_config)
    state_dict = torch.load("your_model_path/videochat2_7b_stage3.pth", "cpu")
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model = model.eval()

    chat = Chat(model)
    print('Initialization Finished')
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img(gr_img, gr_video, chat_state, num_segments):
    print(gr_img, gr_video)
    chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []
    if gr_img is None and gr_video is None:
        return None, None, gr.update(interactive=True),gr.update(interactive=True, placeholder='Please upload video/image first!'), chat_state, None
    if gr_video: 
        llm_message, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, num_segments)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list
    if gr_img:
        llm_message, img_list,chat_state = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state =  chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message,llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state, img_list


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

title = """<h1 align="center"><a href="https://github.com/OpenGVLab/Ask-Anything"><img src="https://s1.ax1x.com/2023/05/07/p9dBMOU.png" alt="Ask-Anything" border="0" style="margin: 0 auto; height: 100px;" /></a> </h1>"""
description ="""
        VideoChat2 powered by InternVideo!<br><p><a href='https://github.com/OpenGVLab/Ask-Anything'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """


with gr.Blocks(title="InternVideo-VideoChat!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                with gr.Tab("Video", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload").style(height=360)
                with gr.Tab("Image", elem_id='image_tab'):
                    up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload").style(height=360)
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            
            num_segments = gr.Slider(
                minimum=8,
                maximum=64,
                value=8,
                step=1,
                interactive=True,
                label="Video Segments",
            )
        
        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")     
    
    chat = init_model()
    upload_button.click(upload_img, [up_image, up_video, chat_state, num_segments], [up_image, up_video, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, up_image, up_video, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True)
# demo.launch(server_name="0.0.0.0", server_port=10034, enable_queue=True)
