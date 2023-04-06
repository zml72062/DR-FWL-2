"""
Utils file for training.
"""

import time
import os
from datetime import datetime
from typing import Optional, Dict
import json
from collections.abc import MutableMapping


def get_exp_name(loader: Dict) -> str:
    r"""Get experiment name.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    arg_list = [loader.dataset.root.split("/")[-1]]
    arg_list.append(str(loader.dataset.target))
    exp_name = "_".join(arg_list)
    return exp_name + f"-{time.strftime('%Y%m%d%H%M%S')}"



def copy(config_path: str, save_dir: str,
         copy_data: bool = False,
         data_path: Optional[str] = None) -> str:
    now = datetime.now()
    dt = now.strftime("%Y-%m-%d-%H-%M-%S")
    dir = os.path.join(save_dir, dt)
    os.system(f"mkdir -p {dir}")
    os.system(f"cp {config_path} {dir}")
    if copy_data:
        raw_dir = os.path.join(data_path, 'raw')
        os.system(f"cp -r {raw_dir} {dir}")
    return dir


class Loader:
    def __init__(self, d: dict):
        for key in d:
            if isinstance(d[key], dict):
                self.__setattr__(key, Loader(d[key]))
            else:
                self.__setattr__(key, d[key])



def load_json(file: str):
    with open(file) as f:
        f = json.load(f)
    return f

def json_loader(f: dict):
    return Loader(f)


def get_seed(seed=234):
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = seed + ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed

