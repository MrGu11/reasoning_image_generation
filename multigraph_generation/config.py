from typing import List, Optional, Sequence, Tuple, TypeAlias, Union, Dict

BoundsType: TypeAlias = Union[Tuple[float, float], Tuple[float, float, float, float]]

# ---------------------------
# 配置集中管理类（便于扩展修改）
# ---------------------------
class Config:
    """几何生成器配置常量，统一管理可配置参数"""
    # 颜色配置
    COLOR_PALETTES = {
        "vibrant": ["#FF3366", "#3366FF", "#33CC99", "#FFCC00", "#9966FF", "#FF6666"],
        "muted": ["#88A0A8", "#C9B1BD", "#D6E0DF", "#F1E9DA", "#B8A9C9"],
    }
    LINE_STYLES = ["-"]
    FILL_PATTERNS = ["", "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    GRADIENT_COLORS = [
        ("#FF6B6B", "#4ECDC4"), ("#45B7D1", "#FFA07A"), ("#98D8C8", "#F0E68C"),
        ("#FF9966", "#FF5E62"), ("#6A85B6", "#BAC8E0"),
    ]
    
    # 默认参数
    DEFAULT_BOUNDS: BoundsType = (-5.0, 5.0)
    DEFAULT_GLOBAL_SCALE = 1.3
    DEFAULT_DPI = 300
    DEFAULT_SHAPE_COUNT_RANGE = (1, 5)
    DEFAULT_GENERATE_ATTEMPTS = 60
