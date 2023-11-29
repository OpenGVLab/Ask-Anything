import torch
import torch.nn as nn
import logging


from .Qformer import BertConfig, BertLMHeadModel
from models.utils import load_temp_embed_with_mismatch

logger = logging.getLogger(__name__)


def build_qformer(num_query_token, vision_width, 
                  qformer_hidden_dropout_prob=0.1,
                  qformer_attention_probs_dropout_prob=0.1,
                  drop_path_rate=0.,
                  ):
    encoder_config = BertConfig.from_pretrained("bert-base-uncased", local_files_only=True)
    encoder_config.encoder_width = vision_width
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = 2
    encoder_config.query_length = num_query_token
    encoder_config.hidden_dropout_prob = qformer_hidden_dropout_prob
    encoder_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob
    encoder_config.drop_path_list = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_config.num_hidden_layers)]
    logger.info(f"Drop_path:{encoder_config.drop_path_list}")
    logger.info(encoder_config)
    Qformer = BertLMHeadModel.from_pretrained(
        "bert-base-uncased", config=encoder_config, local_files_only=True
    )                 
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens

def interpolate_pos_embed_blip(state_dict, new_model):
    if "vision_temp_embed" in state_dict:
        vision_temp_embed_new = new_model.state_dict()["vision_temp_embed"]
        state_dict["vision_temp_embed"] = load_temp_embed_with_mismatch(
            state_dict["vision_temp_embed"], vision_temp_embed_new, add_zero=False
        )
    return state_dict
