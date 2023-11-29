from .xbert import BertConfig, BertForMaskedLM, BertLMHeadModel, BertModel

import logging
logger = logging.getLogger(__name__)

def build_bert(model_config, pretrain, checkpoint):
    """build text encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO

    """
    bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
    bert_config.encoder_width = model_config.vision_encoder.d_model
    bert_config.gradient_checkpointing = checkpoint
    bert_config.fusion_layer = model_config.text_encoder.fusion_layer

    if not model_config.multimodal.enable:
        bert_config.fusion_layer = bert_config.num_hidden_layers

    if pretrain:
        text_encoder, loading_info = BertForMaskedLM.from_pretrained(
            model_config.text_encoder.pretrained,
            config=bert_config,
            output_loading_info=True,
        )
    else:
        text_encoder, loading_info = BertModel.from_pretrained(
            model_config.text_encoder.pretrained,
            config=bert_config,
            add_pooling_layer=False,
            output_loading_info=True,
        )

    return text_encoder


def build_bert_decoder(model_config, checkpoint):
    """build text decoder the same as the multimodal encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO

    """
    bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
    bert_config.encoder_width = model_config.vision_encoder.d_model
    bert_config.gradient_checkpointing = checkpoint

    bert_config.fusion_layer = 0
    bert_config.num_hidden_layers = (
        bert_config.num_hidden_layers - model_config.text_encoder.fusion_layer
    )

    text_decoder, loading_info = BertLMHeadModel.from_pretrained(
        model_config.text_encoder.pretrained,
        config=bert_config,
        output_loading_info=True,
    )

    return text_decoder
