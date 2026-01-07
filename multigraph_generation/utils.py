import logging
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
    RegularPolygon,
    Wedge,
    PathPatch
)
from typing import List, Optional, Sequence, Tuple, TypeAlias, Union, Dict
import numpy as np
import random
from matplotlib import transforms as mtransforms
import math
from matplotlib.transforms import Bbox
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, Collection

from parameter import ShapeParameters

# ---------------------------
# 类型别名定义（简化复杂类型标注）
# ---------------------------
BoundsType: TypeAlias = Union[Tuple[float, float], Tuple[float, float, float, float]]
PointType: TypeAlias = Tuple[float, float]
BboxType: TypeAlias = Tuple[float, float, float, float]


# ---------------------------
# 形状工具类（通用操作封装）
# ---------------------------
class ShapeUtils:
    """提供形状中心、边界、平移等通用工具方法"""

    @staticmethod
    def get_bbox(shape, ax: plt.Axes) -> Bbox:
        """
        获取各种 Patch 或 Line2D 在 data 坐标系下的边界框。

        支持类型：
        Circle, Ellipse, Rectangle, Polygon, RegularPolygon, Wedge, FancyBboxPatch, Line2D
        以及通用 Patch。

        Args:
            shape: Matplotlib 的 Patch 或 Line2D 对象
            ax (plt.Axes): 对应的 Axes 对象

        Returns:
            Bbox: 形状在 data 坐标系下的边界框
        """
        # Rectangle
        if isinstance(shape, Rectangle):
            x0, y0 = shape.get_xy()
            x1 = x0 + shape.get_width()
            y1 = y0 + shape.get_height()
            return Bbox.from_extents(x0, y0, x1, y1)

        # Circle
        elif isinstance(shape, Circle):
            cx, cy = shape.center
            r = shape.radius
            return Bbox.from_extents(cx - r, cy - r, cx + r, cy + r)

        # Ellipse
        elif isinstance(shape, Ellipse):
            cx, cy = shape.center
            w, h = shape.width, shape.height
            return Bbox.from_extents(cx - w/2, cy - h/2, cx + w/2, cy + h/2)

        # Polygon
        elif isinstance(shape, Polygon):
            verts = shape.get_xy()
            x0, y0 = np.min(verts[:, 0]), np.min(verts[:, 1])
            x1, y1 = np.max(verts[:, 0]), np.max(verts[:, 1])
            return Bbox.from_extents(x0, y0, x1, y1)

        # RegularPolygon
        elif isinstance(shape, RegularPolygon):
            # 使用 get_patch_transform() 得到只在 data 坐标下的变换
            transformed_verts = shape.get_path().transformed(shape.get_patch_transform()).vertices
            x0, y0 = np.min(transformed_verts[:, 0]), np.min(transformed_verts[:, 1])
            x1, y1 = np.max(transformed_verts[:, 0]), np.max(transformed_verts[:, 1])
            return Bbox.from_extents(x0, y0, x1, y1)

        # Wedge
        elif isinstance(shape, Wedge):
            # 基本参数（theta 单位转为弧度）
            theta1 = math.radians(float(shape.theta1)) % (2 * math.pi)
            theta2 = math.radians(float(shape.theta2)) % (2 * math.pi)
            r_outer = float(shape.r)
            # 如果存在 width，则 inner radius = r - width（width 可能为 None）
            width = getattr(shape, "width", None)
            if width is None:
                r_inner = 0.0
            else:
                r_inner = max(0.0, r_outer - float(width))

            cx, cy = shape.center  # center 在 data 坐标

            # 判断角度 angle 是否在从 theta1 -> theta2 的弧段内（包含边界），支持跨 2π 情况
            def _angle_in_arc(angle, a1, a2, eps=1e-12):
                angle = angle % (2 * math.pi)
                a1 = a1 % (2 * math.pi)
                a2 = a2 % (2 * math.pi)
                if a1 <= a2:
                    return (a1 - eps) <= angle <= (a2 + eps)
                else:
                    # 跨越 2π: [a1, 2π) U [0, a2]
                    return angle >= (a1 - eps) or angle <= (a2 + eps)

            # 关键角：扇弧端点 + 0, pi/2, pi, 3pi/2（可能是 x/y 极值）
            critical_angles = [theta1, theta2, 0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]

            pts = []
            for ang in critical_angles:
                if _angle_in_arc(ang, theta1, theta2):
                    # 外弧上的点
                    x = cx + r_outer * math.cos(ang)
                    y = cy + r_outer * math.sin(ang)
                    pts.append((x, y))
                    # 若有内半径且 > 0，则内弧同角也需考虑（环状扇形）
                    if r_inner > 0.0:
                        xi = cx + r_inner * math.cos(ang)
                        yi = cy + r_inner * math.sin(ang)
                        pts.append((xi, yi))

            # 同时要把弧端点在两端的径向端考虑（有时弧端的内外端会给出 bbox 边界）
            for ang in (theta1, theta2):
                # 外端
                x = cx + r_outer * math.cos(ang)
                y = cy + r_outer * math.sin(ang)
                pts.append((x, y))
                # 内端（如果有）
                if r_inner > 0.0:
                    xi = cx + r_inner * math.cos(ang)
                    yi = cy + r_inner * math.sin(ang)
                    pts.append((xi, yi))

            # 如果扇形是实心从中心到外弧（r_inner == 0），还应考虑中心点（虽然通常 center 在内部不会影响 bbox）
            if r_inner <= 0.0:
                pts.append((cx, cy))

            # 防御性：如果没有采到点（极少见），回退到外接圆近似
            if not pts:
                x0, y0 = cx - r_outer, cy - r_outer
                x1, y1 = cx + r_outer, cy + r_outer
                return Bbox.from_extents(x0, y0, x1, y1)

            pts = np.asarray(pts, dtype=np.float64)
            x0, y0 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
            x1, y1 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
            return Bbox.from_extents(x0, y0, x1, y1)

        # FancyBboxPatch
        elif isinstance(shape, FancyBboxPatch):
            x0, y0 = shape.get_x(), shape.get_y()
            x1 = x0 + shape.get_width()
            y1 = y0 + shape.get_height()
            return Bbox.from_extents(x0, y0, x1, y1)

        # Line2D
        elif isinstance(shape, Line2D):
            xdata = shape.get_xdata(orig=False)
            ydata = shape.get_ydata(orig=False)
            x0, y0 = np.min(xdata), np.min(ydata)
            x1, y1 = np.max(xdata), np.max(ydata)
            return Bbox.from_extents(x0, y0, x1, y1)

        # PathPatch (更精确地用 path + patch_transform 计算 data 坐标 bbox)
        elif isinstance(shape, PathPatch):
            path = shape.get_path()
            # get_patch_transform(): shape local -> data
            t_patch_to_data = shape.get_patch_transform()
            raw = np.asarray(path.vertices, dtype=np.float64)
            if raw.size == 0:
                raise ValueError("empty path.vertices")

            # 将顶点变换到 data 坐标
            verts_data = t_patch_to_data.transform(raw)

            # 有些 path 的最后一点与第一点重复（闭合），去掉重复点以免影响计算
            if len(verts_data) > 1 and np.allclose(verts_data[0], verts_data[-1]):
                verts_data = verts_data[:-1]

            # 如果变换后仍无有效顶点，抛出异常以触发回退逻辑
            if verts_data.size == 0 or verts_data.shape[0] < 1:
                raise ValueError("no valid vertices after transform")

            x0, y0 = float(np.min(verts_data[:, 0])), float(np.min(verts_data[:, 1]))
            x1, y1 = float(np.max(verts_data[:, 0])), float(np.max(verts_data[:, 1]))

            # 可选：考虑 linewidth 导致的视觉扩张（以 display 像素为单位再转换回 data）
            # 若不需要精确到像素，可以注释掉下面的 padding 相关代码
            try:
                lw = float(getattr(shape, "get_linewidth", lambda: 0.0)())
                if lw > 0:
                    # linewidth 单位为 points；转换为像素： px = points * dpi / 72
                    dpi = ax.figure.dpi
                    pad_px = lw * dpi / 72.0
                    # 从显示坐标向 data 坐标映射一个水平偏移 pad_px，然后用差值作为 data padding
                    # 选择参考点为 bbox 的中心（显示坐标）
                    bbox_disp = Bbox.from_extents(x0, y0, x1, y1).transformed(ax.transData)
                    cx_disp, cy_disp = (bbox_disp.x0 + bbox_disp.x1) / 2.0, (bbox_disp.y0 + bbox_disp.y1) / 2.0
                    p1 = np.array([cx_disp, cy_disp])
                    p2 = np.array([cx_disp + pad_px, cy_disp])
                    try:
                        p1_data = ax.transData.inverted().transform_point(p1)
                        p2_data = ax.transData.inverted().transform_point(p2)
                        pad_data_x = abs(p2_data[0] - p1_data[0])
                        # 对 y 也做类似估计（保守做法：用相同 pad）
                        pad_data_y = abs(p2_data[1] - p1_data[1]) if False else pad_data_x
                        x0 -= pad_data_x
                        x1 += pad_data_x
                        y0 -= pad_data_y
                        y1 += pad_data_y
                    except Exception:
                        # 如果转换失败，忽略 padding
                        pass
            except Exception:
                # 忽略 padding 计算中的任何错误
                pass
            return Bbox.from_extents(x0, y0, x1, y1)
                # 其他 Patch 类型：使用 display 坐标转换
                
        elif isinstance(shape, Patch):
            bbox_disp = shape.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            return bbox_disp.transformed(ax.transData.inverted())

        else:
            raise TypeError(f"Unsupported shape type: {type(shape)}")

    @staticmethod
    def get_center(shape: Patch, ax: plt.Axes):
        """
        返回 shape 在 ax 的 data 坐标系下的中心 (cx, cy)。
        采用 get_bbox 的中心作为统一接口（适用于任意 patch）。
        """
        bbox = ShapeUtils.get_bbox(shape, ax)
        cx = (bbox.x0 + bbox.x1) / 2.0
        cy = (bbox.y0 + bbox.y1) / 2.0
        return cx, cy

    @staticmethod
    def translate_shape(shape, dx: float, dy: float, ax: plt.Axes) -> None:
        """
        在 data 坐标系中平移 shape：(dx, dy) 的单位为 ax 的数据坐标单位。
        优先尝试以 data 坐标修改 artist 的数据/属性；仅在无法做到时退回到 display-space transform 叠加。
        注意：此函数**不**强制触发重绘（调用者可在最后调用 fig.canvas.draw_idle()/draw()）。
        """
        # helper: convert array-like safely
        def _to_array(x):
            try:
                return np.asarray(x, dtype=np.float64)
            except Exception:
                return None

        # 1) Line2D: 直接修改 x/y data（这是移动线段的正确方式）
        try:
            if isinstance(shape, Line2D):
                x = _to_array(shape.get_xdata())
                y = _to_array(shape.get_ydata())
                if x is None or y is None:
                    raise RuntimeError("Line2D data not array-like")
                # 保持原始形状（mask/nan 保留）
                shape.set_xdata(x + dx)
                shape.set_ydata(y + dy)
                return

            # 2) PathCollection / Collection（如 scatter）：尝试修改 offsets（data 坐标）
            if isinstance(shape, PathCollection) or isinstance(shape, Collection):
                # 多数 Collection（scatter）支持 get_offsets / set_offsets
                if hasattr(shape, "get_offsets") and hasattr(shape, "set_offsets"):
                    offsets = shape.get_offsets()
                    # offsets 可能是空或非数组结构
                    offs = _to_array(offsets)
                    if offs is None or offs.size == 0:
                        # 无偏移可修改，抛出以进入 fallback
                        raise RuntimeError("Collection offsets empty or non-array")
                    # offs shape: (N, 2) 或 (N,). 处理 1-D -> 2-D 判断
                    if offs.ndim == 1 and offs.size == 2:
                        new_offs = offs + np.array([dx, dy], dtype=np.float64)
                    else:
                        new_offs = offs + np.array([dx, dy], dtype=np.float64)
                    shape.set_offsets(new_offs)
                    return
                # 某些 Collection 无 offsets，可继续尝试其它属性或 fallback
                # 继续执行下方其它 cases / 最终 fallback

            # 3) Circle / Ellipse / Wedge: 修改 center（data 坐标）
            if isinstance(shape, (Circle, Ellipse, Wedge)):
                # 使用 get_center/ set_center 优先
                if hasattr(shape, "get_center"):
                    cx, cy = shape.get_center()
                else:
                    # 有些实现用 center 属性直接存储
                    cx, cy = getattr(shape, "center", (None, None))
                    if cx is None:
                        # 尝试通用 bbox 中心
                        cx, cy = ShapeUtils.get_center(shape, ax)
                new_center = (cx + dx, cy + dy)
                if hasattr(shape, "set_center"):
                    shape.set_center(new_center)
                else:
                    try:
                        shape.center = new_center
                    except Exception:
                        raise
                return

            # 4) Rectangle / FancyBboxPatch: 修改 x/y（data 坐标）
            if isinstance(shape, (Rectangle, FancyBboxPatch)):
                # Rectangle 提供 get_x/get_y/set_x/set_y
                try:
                    shape.set_x(shape.get_x() + dx)
                    shape.set_y(shape.get_y() + dy)
                    return
                except Exception:
                    # fall through to fallback
                    raise

            # 5) Polygon: 修改顶点（data 坐标）
            if isinstance(shape, Polygon):
                verts = _to_array(shape.get_xy())
                if verts is None:
                    raise RuntimeError("Polygon vertices not accessible")
                shape.set_xy(verts + np.array([dx, dy], dtype=np.float64))
                return

            # 6) RegularPolygon: 尝试通过中心或 xy 修改
            if isinstance(shape, RegularPolygon):
                # 尝试常见接口
                if hasattr(shape, "get_xy"):
                    try:
                        # some RegularPolygon expose center as .xy or via get_xy
                        cur = shape.get_xy()
                        # if get_xy gives vertices, fallback to bbox center
                        if isinstance(cur, tuple) and len(cur) == 2:
                            cx, cy = cur
                        else:
                            cx, cy = ShapeUtils.get_center(shape, ax)
                    except Exception:
                        cx, cy = ShapeUtils.get_center(shape, ax)
                elif hasattr(shape, "xy"):
                    try:
                        cx, cy = shape.xy
                    except Exception:
                        cx, cy = ShapeUtils.get_center(shape, ax)
                else:
                    cx, cy = ShapeUtils.get_center(shape, ax)
                # 尝试修改可能存在的属性
                if hasattr(shape, "set_xy"):
                    try:
                        shape.set_xy((cx + dx, cy + dy))
                        return
                    except Exception:
                        pass
                try:
                    shape.xy = (cx + dx, cy + dy)
                    return
                except Exception:
                    # fallback below
                    raise

            # 7) 其他 Patch: 尝试通用 get_xy/set_xy
            if isinstance(shape, Patch):
                if hasattr(shape, "get_xy") and hasattr(shape, "set_xy"):
                    verts = _to_array(shape.get_xy())
                    if verts is None:
                        raise RuntimeError("Patch vertices not array-like")
                    shape.set_xy(verts + np.array([dx, dy], dtype=np.float64))
                    return

        except Exception as e:
            # 如果上面任何直接 data 修改抛出异常，就走 fallback（display-space transform）
            logging.warning("translate_shape 最终 fallback 失败：%s", e)
            pass

    @staticmethod
    def _reposition_regular_polygon(shape: RegularPolygon, new_center: PointType, radius: float) -> None:
        """重新定位正多边形（兼容不同matplotlib版本）"""
        try:
            shape.xy = new_center

        except Exception as e:
            logging.debug("重新定位正多边形失败：%s，使用平移fallback", e)
            # 平移fallback
            cx_old, cy_old = ShapeUtils.get_center(shape)
            ShapeUtils.translate_shape(shape, new_center[0] - cx_old, new_center[1] - cy_old)

    @staticmethod
    def random_point(xy_range: BoundsType = (-4.0, 4.0)) -> PointType:
        """生成范围内的随机点（支持2/4元组范围）"""
        if len(xy_range) == 2:
            lo, hi = xy_range
            return (random.uniform(lo, hi), random.uniform(lo, hi))
        x0, x1, y0, y1 = xy_range
        return (random.uniform(x0, x1), random.uniform(y0, y1))

    @staticmethod
    def random_size(min_size: float = 1.2, max_size: float = 3.2) -> float:
        """生成随机尺寸"""
        return float(random.uniform(min_size, max_size))

    @staticmethod
    def random_rotation() -> float:
        """生成随机旋转角度（-180到180度）"""
        return float(random.uniform(-180.0, 180.0))

    @staticmethod
    def check_bounds(shape: Patch, ax: plt.Axes, bounds: BoundsType = (-5.0, 5.0)) -> bool:
        """检查形状是否完全在边界内"""
        # 解析边界
        if len(bounds) == 2:
            x_min, x_max = bounds
            y_min, y_max = bounds
        else:
            x_min, x_max, y_min, y_max = bounds

        # 检查边界框
        try:
            x0, y0, x1, y1 = ShapeUtils.get_bbox(shape)
            return (x0 >= x_min and x1 <= x_max and y0 >= y_min and y1 <= y_max)
        except Exception as e:
            # 边界框检查失败，用中心检查
            c = ShapeUtils.get_center(shape, ax)
            logging.debug("边界框检查失败，使用中心检查：%s", e)
            return (c[0] >= x_min and c[0] <= x_max and c[1] >= y_min and c[1] <= y_max)

    @staticmethod
    def get_shape_parameters(shape: Patch, ax: plt.Axes, shape_id: str) -> ShapeParameters:
        """提取形状的参数信息"""
        shape_type = type(shape).__name__
        center = ShapeUtils.get_center(shape, ax)
        bbox = ShapeUtils.get_bbox(shape, ax)
        
        # 提取尺寸信息
        size = 0.0
        if isinstance(shape, (Circle, Wedge)):
            size = float(getattr(shape, "radius", getattr(shape, "r", 0.0)))
        elif isinstance(shape, Ellipse):
            size = (float(shape.width), float(shape.height))
        elif isinstance(shape, (Rectangle, FancyBboxPatch)):
            size = (float(shape.get_width()), float(shape.get_height()))
        elif isinstance(shape, RegularPolygon):
            size = float(getattr(shape, "radius", getattr(shape, "r", 0.0)))
        elif isinstance(shape, Polygon):
            x0, y0, x1, y1 = bbox
            size = (x1 - x0, y1 - y0)
        
        # 提取旋转角度
        rotation = 0.0
        if hasattr(shape, "angle"):
            rotation = float(shape.angle)
        elif isinstance(shape, RegularPolygon) and hasattr(shape, "orientation"):
            rotation = math.degrees(float(shape.orientation))
        
        # 提取样式信息
        edge_color = getattr(shape, "get_edgecolor", lambda: None)()
        if isinstance(edge_color, (tuple, np.ndarray)):
            edge_color = mpl.colors.to_hex(edge_color)
            
        line_width = getattr(shape, "get_linewidth", lambda: None)()
        line_style = getattr(shape, "get_linestyle", lambda: None)()
        
        fill_color = getattr(shape, "get_facecolor", lambda: None)()
        if isinstance(fill_color, (tuple, np.ndarray)):
            fill_color = mpl.colors.to_hex(fill_color)
            
        alpha = getattr(shape, "get_alpha", lambda: None)()

        # 创建参数对象
        params = ShapeParameters(
            shape_id=shape_id,
            shape_type=shape_type,
            center=center,
            bbox=bbox,
            size=size,
            rotation=rotation,
            edge_color=edge_color,
            line_width=float(line_width) if line_width is not None else None,
            line_style=str(line_style) if line_style is not None else None,
            fill_color=fill_color,
            alpha=float(alpha) if alpha is not None else None
        )
        
        # 添加特定类型的额外参数
        if isinstance(shape, Wedge):
            params.extra_params = {
                "theta1": getattr(shape, "theta1", 0.0),
                "theta2": getattr(shape, "theta2", 360.0)
            }
        elif isinstance(shape, RegularPolygon):
            params.extra_params = {
                "num_vertices": getattr(shape, "numVertices", getattr(shape, "N", 3))
            }
        elif isinstance(shape, FancyBboxPatch):
            params.extra_params = {
                "boxstyle": str(getattr(shape, "get_boxstyle", lambda: "")())
            }
            
        return params
    
    @staticmethod
    def is_point_inside_shape(shape: Patch, ax: plt.Axes, point: Tuple[float, float]) -> bool:
        """
        判断点（数据坐标）是否在形状内部
        支持 Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle, RegularPolygon, Wedge
        """
        x, y = point

        # 1️⃣ 对 Circle, Ellipse, RegularPolygon, Wedge 可用解析法
        if isinstance(shape, Circle):
            cx, cy = shape.center
            r = shape.radius
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

        elif isinstance(shape, Ellipse):
            cx, cy = shape.center
            a, b = shape.width / 2.0, shape.height / 2.0
            phi = np.deg2rad(getattr(shape, "angle", 0.0))
            # 将点旋转到椭圆主轴
            xp = np.cos(phi) * (x - cx) + np.sin(phi) * (y - cy)
            yp = -np.sin(phi) * (x - cx) + np.cos(phi) * (y - cy)
            return (xp / a) ** 2 + (yp / b) ** 2 <= 1.0

        elif isinstance(shape, RegularPolygon):
            # 转为多边形顶点求解
            verts = shape.get_path().vertices @ shape.get_transform().get_matrix()[:2, :2].T
            cx, cy = shape.xy
            verts += np.array([cx, cy])
            path = shape.get_path().transformed(shape.get_transform())
            return path.contains_point((x, y))

        elif isinstance(shape, Wedge):
            # 判断半径 + 角度
            cx, cy = shape.center
            r = shape.r
            theta1, theta2 = np.deg2rad(shape.theta1), np.deg2rad(shape.theta2)
            dx, dy = x - cx, y - cy
            dist2 = dx ** 2 + dy ** 2
            if dist2 > r ** 2:
                return False
            angle = np.arctan2(dy, dx) % (2 * np.pi)
            t1, t2 = theta1 % (2*np.pi), theta2 % (2*np.pi)
            if t1 <= t2:
                return t1 <= angle <= t2
            else:
                # 跨 2π 场景
                return angle >= t1 or angle <= t2

        # 2️⃣ 对 Polygon, Rectangle, FancyBboxPatch, Patch 等通用 path
        else:
            try:
                path = shape.get_path().transformed(shape.get_transform())
                return path.contains_point((x, y))
            except Exception:
                # fallback: 返回 False
                return False