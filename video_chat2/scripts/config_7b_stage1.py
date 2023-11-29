from configs.data import *
from configs.model import *

# ========================= data ==========================
train_corpus = "webvid10m_cc14m"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict(msrvtt_1k_test=available_corpus["msrvtt_1k_test"])
test_types = ["msrvtt_1k_test"]

num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 128
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
text_enc = "bert"
model = dict(
    model_cls="VideoChat2_qformer",
    vision_encoder=dict(
        name="vit_l14",
        img_size=224, 
        patch_size=16, 
        d_model=1024,
        encoder_embed_dim=1024, 
        encoder_depth=24,
        encoder_num_heads=16, 
        drop_path_rate=0., 
        num_frames="${num_frames}",
        tubelet_size=1,
        use_checkpoint=False,
        checkpoint_num=12,
        pretrained="your_model_path/l16_25m.pth",
        return_index=-2,
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    vit_add_ln=True,
    embed_dim=768,
    temp=0.07,
    qformer_num_query_tokens=32,
    agg_method="mean",
    drop_path_rate=0.2,
)

criterion = dict(
    loss_weight=dict(vtc=1.0, mlm=0.0, vtm=1.0, mvm=0.0, cap=1.0),  # 0: disabled.
    vtm_hard_neg=True,
    vtm_cat_text_cls=True
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

scheduler = dict(sched="cosine", epochs=10, min_lr_multi=0.01, warmup_epochs=0.2)

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
    project="videochat2",  # setup in your command line
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
