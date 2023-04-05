"""
save_result.py - Copy configure files.
"""
import os
from datetime import datetime
from typing import Optional


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