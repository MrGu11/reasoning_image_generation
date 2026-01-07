from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, TypeAlias, Union, Dict

PointType: TypeAlias = Tuple[float, float]
BboxType: TypeAlias = Tuple[float, float, float, float]

# ---------------------------
# 图形参数数据类
# ---------------------------
@dataclass
class ShapeParameters:
    """存储单个图形的所有参数信息"""
    shape_id: str  # 唯一标识符
    shape_type: str  # 图形类型
    center: PointType  # 中心坐标
    bbox: BboxType  # 边界框
    size: Union[float, Tuple[float, float]]  # 尺寸（半径或宽高）
    rotation: float = 0.0  # 旋转角度（度）
    edge_color: Optional[str] = None  # 边缘颜色
    line_width: Optional[float] = None  # 线宽
    line_style: Optional[str] = None  # 线型
    fill_color: Optional[str] = None  # 填充颜色
    alpha: Optional[float] = None  # 透明度
    has_gradient: bool = False  # 是否有渐变
    gradient_colors: Optional[Tuple[str, str]] = None  # 渐变颜色
    has_mask: bool = False  # 是否有掩码
    mask_type: Optional[str] = None  # 掩码类型
    has_decoration: bool = False  # 是否有装饰
    decoration_style: Optional[str] = None  # 装饰样式
    extra_params: dict = field(default_factory=dict)  # 额外参数