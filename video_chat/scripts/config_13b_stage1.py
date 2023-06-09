from configs.data import *
# ========================= data ==========================
train_corpus = "webvid10m_14m"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict()
test_types = []
num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 16
max_txt_l = 32

pre_text = False

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
model = dict(
    vit_model="eva_clip_g",
    vit_model_path="model/eva_vit_g.pth",
    q_former_model_path="model/blip2_pretrained_flant5xxl.pth",
    llama_model_path="model/stable-vicuna-13b",
    gpt_model_path="",
    img_size=224,
    num_query_token=32,
    drop_path_rate=0.,
    use_grad_checkpoint=False,
    vit_precision="fp32",
    freeze_vit=True,
    freeze_mhra=False, # open mhra
    freeze_qformer=True,
    low_resource=False,
    prompt_path="prompts/concise_description.txt",
    img_prompt_path="prompts/concise_image_description.txt",
    prompt_template="###Human: {} ###Assistant: ",
    max_txt_len="${max_txt_l}", # use large max_txt_len on stage2
    end_sym="###",
    # uniformerv2
    temporal_downsample=False,
    no_lmhra=True,
    double_lmhra=False,
    lmhra_reduction=2.0,
    gmhra_layers=8,
    gmhra_drop_path_rate=0.,
    gmhra_dropout=0.5,
    # qformer
    extra_num_query_token=64,
)

optimizer = dict(
    opt="adamW",
    lr=1e-4,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=1, min_lr_multi=0.01, warmup_epochs=0.2)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="user",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="videochat",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = True
auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?
