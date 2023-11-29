from __future__ import annotations

import argparse
import ast
import json
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from importlib import import_module

import yaml

from .easydict import EasyDict

__all__ = ["Config", "pretty_text"]


BASE_KEY = "_base_"
# BASE_CONFIG = {"OUTPUT_DIR": "./workspace", "SESSION": "base", "LOG_FILE": "log.txt"}
BASE_CONFIG = {}

cfg = None


class Config(object):
    """config"""

    @classmethod
    def pretty_text(cls, cfg: dict, indent=2) -> str:
        """format dict to a string

        Args:
            cfg (EasyDict): the params.

        Returns: The string to display.

        """
        msg = "{\n"
        for i, (k, v) in enumerate(cfg.items()):
            if isinstance(v, dict):
                v = cls.pretty_text(v, indent + 4)
            spaces = " " * indent
            msg += spaces + "{}: {}".format(k, v)
            if i == len(cfg) - 1:
                msg += " }"
            else:
                msg += "\n"
        return msg

    @classmethod
    def dump(cls, cfg, savepath=None):
        """dump cfg to `json` file.

        Args:
            cfg (dict): The dict to dump.
            savepath (str): The filepath to save the dumped dict.

        Returns: TODO

        """
        if savepath is None:
            savepath = osp.join(cfg.WORKSPACE, "config.json")
        json.dump(cfg, open(savepath, "w"), indent=2)

    @classmethod
    def get_config(cls, default_config: dict = None):
        """get a `Config` instance.

        Args:
            default_config (dict): The default config. `default_config` will be overrided
                by config file `--cfg`, `--cfg` will be overrided by commandline args.

        Returns: an EasyDict.
        """
        global cfg
        if cfg is not None:
            return cfg

        # define arg parser.
        parser = argparse.ArgumentParser()
        # parser.add_argument("--cfg", help="load configs from yaml file", default="", type=str)
        parser.add_argument(
            "config_file", help="the configuration file to load. support: .yaml, .json, .py"
        )
        parser.add_argument(
            "opts",
            default=None,
            nargs="*",
            help="overrided configs. List. Format: 'key1 name1 key2 name2'",
        )
        args = parser.parse_args()

        cfg = EasyDict(BASE_CONFIG)
        if osp.isfile(args.config_file):
            cfg_from_file = cls.from_file(args.config_file)
            cfg = merge_a_into_b(cfg_from_file, cfg)
        cfg = cls.merge_list(cfg, args.opts)
        cfg = eval_dict_leaf(cfg)

        # update some keys to make them show at the last
        for k in BASE_CONFIG:
            cfg[k] = cfg.pop(k)
        return cfg

    @classmethod
    def from_file(cls, filepath: str) -> EasyDict:
        """Build config from file. Supported filetypes: `.py`,`.yaml`,`.json`.

        Args:
            filepath (str): The config file path.

        Returns: TODO

        """
        filepath = osp.abspath(osp.expanduser(filepath))
        if not osp.isfile(filepath):
            raise IOError(f"File does not exist: {filepath}")
        if filepath.endswith(".py"):
            with tempfile.TemporaryDirectory() as temp_config_dir:

                shutil.copytree(osp.dirname(filepath), osp.join(temp_config_dir, "tmp_config"))
                sys.path.insert(0, temp_config_dir)
                mod = import_module("tmp_config." + osp.splitext(osp.basename(filepath))[0])
                # mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                }
                for k in list(sys.modules.keys()):
                    if "tmp_config" in k:
                        del sys.modules[k]
        elif filepath.endswith((".yml", ".yaml")):
            cfg_dict = yaml.load(open(filepath, "r"), Loader=yaml.Loader)
        elif filepath.endswith(".json"):
            cfg_dict = json.load(open(filepath, "r"))
        else:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        cfg_text = filepath + "\n"
        with open(filepath, "r") as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:  # load configs in `BASE_KEY`
            cfg_dir = osp.dirname(filepath)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = (
                base_filename if isinstance(base_filename, list) else [base_filename]
            )

            cfg_dict_list = list()
            for f in base_filename:
                _cfg_dict = Config.from_file(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError("Duplicate key is not allowed among bases")
                base_cfg_dict.update(c)

            cfg_dict = merge_a_into_b(cfg_dict, base_cfg_dict)

        return EasyDict(cfg_dict)

    @classmethod
    def merge_list(cls, cfg, opts: list):
        """merge commandline opts.

        Args:
            cfg: (dict): The config to be merged.
            opts (list): The list to merge. Format: [key1, name1, key2, name2,...].
                The keys can be nested. For example, ["a.b", v] will be considered
                as `dict(a=dict(b=v))`.

        Returns: dict.

        """
        assert len(opts) % 2 == 0, f"length of opts must be even. Got: {opts}"
        for i in range(0, len(opts), 2):
            full_k, v = opts[i], opts[i + 1]
            keys = full_k.split(".")
            sub_d = cfg
            for i, k in enumerate(keys):
                if not hasattr(sub_d, k):
                    raise ValueError(f"The key {k} not exist in the config. Full key:{full_k}")
                if i != len(keys) - 1:
                    sub_d = sub_d[k]
                else:
                    sub_d[k] = v
        return cfg


def merge_a_into_b(a, b, inplace=False):
    """The values in a will override values in b.

    Args:
        a (dict): source dict.
        b (dict): target dict.

    Returns: dict. recursively merge dict a into dict b.

    """
    if not inplace:
        b = deepcopy(b)
    for key in a:
        if key in b:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                b[key] = merge_a_into_b(a[key], b[key], inplace=True)
            else:
                b[key] = a[key]
        else:
            b[key] = a[key]
    return b


def eval_dict_leaf(d, orig_dict=None):
    """eval values of dict leaf.

    Args:
        d (dict): The dict to eval.

    Returns: dict.

    """
    if orig_dict is None:
        orig_dict = d
    for k, v in d.items():
        if not isinstance(v, dict):
            d[k] = eval_string(v, orig_dict)
        else:
            eval_dict_leaf(v, orig_dict)
    return d


def eval_string(string, d):
    """automatically evaluate string to corresponding types.

    For example:
        not a string  -> return the original input
        '0'  -> 0
        '0.2' -> 0.2
        '[0, 1, 2]' -> [0,1,2]
        'eval(1+2)' -> 3
        'eval(range(5))' -> [0,1,2,3,4]
        '${a}' -> d.a



    Args:
        string (str): The value to evaluate.
        d (dict): The

    Returns: the corresponding type

    """
    if not isinstance(string, str):
        return string
    # if len(string) > 1 and string[0] == "[" and string[-1] == "]":
    #     return eval(string)
    if string[0:5] == "eval(":
        return eval(string[5:-1])

    s0 = string
    s1 = re.sub(r"\${(.*)}", r"d.\1", s0)
    if s1 != s0:
        while s1 != s0:
            s0 = s1
            s1 = re.sub(r"\${(.*)}", r"d.\1", s0)
        return eval(s1)

    try:
        v = ast.literal_eval(string)
    except:
        v = string
    return v
