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
import random 
from config import Config
import logging
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms as mtransforms

from utils import ShapeUtils
from parameter import ShapeParameters

# ---------------------------
# 样式增强器
# ---------------------------
class StyleEnhancer:
    """形状样式增强（随机样式、渐变、旋转）"""
    @staticmethod
    def get_random_style(shape: Patch, shape_params: ShapeParameters, 
                        palette: Optional[str] = None, alpha: float = 0.9, line_width: Optional[int] = None) -> Patch:
        """为形状应用随机样式（颜色、边缘、线宽等）并更新参数记录"""
        palette = palette or random.choice(list(Config.COLOR_PALETTES.keys()))
        color = random.choice(Config.COLOR_PALETTES[palette])
        
        # 设置填充色
        try:
            shape.set_facecolor('none')
            shape_params.fill_color = 'none'
        except Exception as e:
            logging.debug("设置%s填充色失败：%s", type(shape), e)

        # 设置边缘
        edge_color = random.choice(["black"])
        try:
            shape.set_edgecolor(edge_color)
            if line_width == None:
                line_width = random.uniform(1.5, 2.0)
            shape.set_linewidth(line_width)
            line_style = random.choice(Config.LINE_STYLES)
            shape.set_linestyle(line_style)
            
            # 更新参数记录
            shape_params.edge_color = edge_color
            shape_params.line_width = line_width
            shape_params.line_style = line_style
        except Exception as e:
            logging.debug("设置%s边缘样式失败：%s", type(shape), e)

        # 设置透明度
        try:
            shape.set_alpha(alpha)
            shape_params.alpha = alpha
        except Exception as e:
            logging.debug("设置%s透明度失败：%s", type(shape), e)

        return shape

    @staticmethod
    def apply_gradient(ax: mpl.axes.Axes, shape: Patch, shape_params: ShapeParameters, 
                      cmap_name: Optional[str] = None, strength: float = 0.75) -> None:
        """为形状应用渐变填充并更新参数记录"""
        if shape not in ax.patches:
            ax.add_patch(shape)

        # 获取形状边界
        try:
            x0, y0, x1, y1 = ShapeUtils.get_bbox(shape)
        except Exception:
            c = ShapeUtils.get_center(shape)
            x0, x1 = c[0] - 1.0, c[0] + 1.0
            y0, y1 = c[1] - 1.0, c[1] + 1.0

        # 创建渐变色图
        c1, c2 = None, None
        if cmap_name is None:
            c1, c2 = random.choice(Config.GRADIENT_COLORS)
            cmap = mpl.colors.LinearSegmentedColormap.from_list("gmap", [c1, c2])
        else:
            cmap = plt.get_cmap(cmap_name)

        # 生成网格数据
        nx, ny = 120, 120
        X = np.linspace(x0, x1, nx, dtype=np.float32)
        Y = np.linspace(y0, y1, ny, dtype=np.float32)
        Xg, Yg = np.meshgrid(X, Y)
        C = np.sqrt((Xg - (x0 + x1)/2)**2 + (Yg - (y0 + y1)/2)** 2)  # 径向渐变

        # 绘制渐变并裁剪到形状
        z_order = getattr(shape, "zorder", 0) - 0.1
        im = ax.imshow(
            C, cmap=cmap, extent=(x0, x1, y0, y1), 
            origin="lower", alpha=strength, zorder=z_order
        )
        try:
            im.set_clip_path(shape, transform=shape.get_transform())
        except Exception as e:
            try:
                im.set_clip_path(shape)
            except Exception as e2:
                logging.debug("裁剪渐变失败：%s", e2)

        shape.set_zorder(z_order + 1)  # 确保形状边缘在渐变上方
        logging.debug("应用渐变：形状类型=%s, 边界=%s", type(shape), (x0, y0, x1, y1))
        
        # 更新参数记录
        shape_params.has_gradient = True
        if c1 and c2:
            shape_params.gradient_colors = (c1, c2)
        shape_params.extra_params["gradient_strength"] = strength

    @staticmethod
    def rotate(shape: Patch, ax: mpl.axes.Axes, shape_params: ShapeParameters, 
              angle: Optional[float] = None) -> None:
        """绕形状中心旋转（支持现有变换组合）并更新参数记录"""
        angle = angle or ShapeUtils.random_rotation()
        cx, cy = ShapeUtils.get_center(shape, ax)

        try:
            # 创建旋转变换
            t = mtransforms.Affine2D().rotate_deg_around(cx, cy, float(angle))
            # 与现有变换组合
            orig_transform = shape.get_transform()
            shape.set_transform(t + orig_transform)
            logging.debug("旋转%s：角度=%.2f, 中心=(%.2f,%.2f)", type(shape), angle, cx, cy)
            
            # 更新参数记录
            shape_params.rotation = angle
        except Exception as e:
            logging.debug("旋转%s失败：%s", type(shape), e)