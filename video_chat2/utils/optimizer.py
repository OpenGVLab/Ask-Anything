""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import re
import torch
from torch import optim as optim
from utils.distributed import is_main_process
import logging
logger = logging.getLogger(__name__)
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay, no_decay_list=(), filter_bias_and_bn=True):
    named_param_tuples = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
            named_param_tuples.append([name, param, 0])
        elif name in no_decay_list:
            named_param_tuples.append([name, param, 0])
        else:
            named_param_tuples.append([name, param, weight_decay])
    return named_param_tuples


def add_different_lr(named_param_tuples_or_model, diff_lr_names, diff_lr, default_lr):
    """use lr=diff_lr for modules named found in diff_lr_names,
    otherwise use lr=default_lr

    Args:
        named_param_tuples_or_model: List([name, param, weight_decay]), or nn.Module
        diff_lr_names: List(str)
        diff_lr: float
        default_lr: float
    Returns:
        named_param_tuples_with_lr: List([name, param, weight_decay, lr])
    """
    named_param_tuples_with_lr = []
    logger.info(f"diff_names: {diff_lr_names}, diff_lr: {diff_lr}")
    for name, p, wd in named_param_tuples_or_model:
        use_diff_lr = False
        for diff_name in diff_lr_names:
            # if diff_name in name:
            if re.search(diff_name, name) is not None:
                logger.info(f"param {name} use different_lr: {diff_lr}")
                use_diff_lr = True
                break

        named_param_tuples_with_lr.append(
            [name, p, wd, diff_lr if use_diff_lr else default_lr]
        )

    if is_main_process():
        for name, _, wd, diff_lr in named_param_tuples_with_lr:
            logger.info(f"param {name}: wd: {wd}, lr: {diff_lr}")

    return named_param_tuples_with_lr


def create_optimizer_params_group(named_param_tuples_with_lr):
    """named_param_tuples_with_lr: List([name, param, weight_decay, lr])"""
    group = {}
    for name, p, wd, lr in named_param_tuples_with_lr:
        if wd not in group:
            group[wd] = {}
        if lr not in group[wd]:
            group[wd][lr] = []
        group[wd][lr].append(p)

    optimizer_params_group = []
    for wd, lr_groups in group.items():
        for lr, p in lr_groups.items():
            optimizer_params_group.append(dict(
                params=p,
                weight_decay=wd,
                lr=lr
            ))
            logger.info(f"optimizer -- lr={lr} wd={wd} len(p)={len(p)}")
    return optimizer_params_group


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # check for modules that requires different lr
    if hasattr(args, "different_lr") and args.different_lr.enable:
        diff_lr_module_names = args.different_lr.module_names
        diff_lr = args.different_lr.lr
    else:
        diff_lr_module_names = []
        diff_lr = None

    no_decay = {}
    if hasattr(model, 'no_weight_decay'):
        no_decay = model.no_weight_decay()
    named_param_tuples = add_weight_decay(
        model, weight_decay, no_decay, filter_bias_and_bn)
    named_param_tuples = add_different_lr(
        named_param_tuples, diff_lr_module_names, diff_lr, args.lr)
    parameters = create_optimizer_params_group(named_param_tuples)

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    return optimizer
