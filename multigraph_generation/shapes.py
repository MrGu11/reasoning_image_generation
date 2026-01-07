from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from typing import List, Optional, Sequence, Tuple, TypeAlias, Union, Dict
import math
import numpy as np
import logging

PointType: TypeAlias = Tuple[float, float]

# ---------------------------
# 基础形状生成器
# ---------------------------
class BaseShapes:
    """基础形状生成工厂（统一创建不同类型形状）"""
    @staticmethod
    def circle(center: PointType = (0.0, 0.0), radius: float = 1.0, **kwargs) -> Circle:
        return Circle(center, radius,** kwargs)

    @staticmethod
    def ellipse(center: PointType = (0.0, 0.0), width: float = 2.0, height: float = 1.0, 
                angle: float = 0.0, **kwargs) -> Ellipse:
        return Ellipse(center, width, height, angle=angle,** kwargs)

    @staticmethod
    def rectangle(xy: PointType = (0.0, 0.0), width: float = 2.0, height: float = 2.0, 
                  round_corner: float = 0.0, **kwargs) -> Patch:
        if round_corner > 0:
            boxstyle = f"round,pad=0,rounding_size={float(round_corner)}"
            return FancyBboxPatch(xy, width, height, boxstyle=boxstyle,** kwargs)
        return Rectangle(xy, width, height, **kwargs)

    @staticmethod
    def polygon(vertices: Sequence[PointType],** kwargs) -> Polygon:
        return Polygon(np.asarray(vertices, dtype=np.float32), closed=True, **kwargs)

    @staticmethod
    def regular_polygon(center: PointType = (0.0, 0.0), num_edges: int = 5, radius: float = 1.0, 
                       orientation: float = 0.0,** kwargs) -> RegularPolygon:
        """创建正多边形（兼容不同matplotlib版本）"""
        try:
            # 优先关键字参数（高版本matplotlib）
            return RegularPolygon(
                xy=center, numVertices=num_edges, radius=radius, orientation=orientation, **kwargs
            )
        except TypeError:
            try:
                # 位置参数（低版本matplotlib）
                return RegularPolygon(center, num_edges, radius, orientation=orientation,** kwargs)
            except Exception as e:
                logging.debug("正多边形直接创建失败，使用多边形模拟：%s", e)
                # 多边形模拟fallback
                verts = []
                for i in range(num_edges):
                    ang = orientation + 2 * math.pi * i / num_edges
                    verts.append((
                        center[0] + radius * math.cos(ang),
                        center[1] + radius * math.sin(ang)
                    ))
                return BaseShapes.polygon(verts, **kwargs)

    @staticmethod
    def sector(center: PointType = (0.0, 0.0), radius: float = 1.0, 
              theta1: float = 0.0, theta2: float = 90.0,** kwargs) -> Wedge:
        return Wedge(center, radius, theta1, theta2, **kwargs)