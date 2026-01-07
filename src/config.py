# config.py
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
import copy

DEFAULT_CATEGORIES = {
    "图形相似": {
        "位置变换": ["平移", "旋转", "翻转(镜像)", "组合"],
        # "位置变换": ["组合"],
        "叠加": ["直接叠加", "去同存异", "去异存同"],
        # "属性": ["封闭开放", "曲直"]
    },
    "图形相异": {
        "图形遍历": ["单一遍历", "位置遍历"],
        # "数量规律": {
        #     "面": ["部分", "图群", "角"],
        #     "线": ["内外线", "直曲线", "平行线", "笔画"],
        #     "点": ["内外点", "交点"]
        # }
    }
}

@dataclass
class GenConfig:
    out_dir: str = './out'
    canvas_size: Tuple[int,int] = (512,512)  # (W,H)
    grid_size: int = 3

    # appearance
    bg_color: Tuple[int,int,int] = (255,255,255)

    # randomness / reproducibility
    seed: int | None = None

    # categories & sampling
    categories: Dict[str,Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CATEGORIES))
    category_weights: Dict[str, float] = field(default_factory=dict)

    # export options
    export_coco: bool = True
    export_json: bool = True

    # ----- new options for sequence reasoning -----
    # minimum and maximum length of the input sequence (number of proto images used in sequence)
    seq_min: int = 2
    seq_max: int = 4

    # number of multiple-choice options shown to user (including the correct one)
    num_options: int = 4

    # whether to shuffle options (True -> options will be shuffled; correct_index stored in meta)
    shuffle_options: bool = True



