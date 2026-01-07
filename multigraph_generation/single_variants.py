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
import random
import matplotlib as mpl
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon, GeometryCollection
from matplotlib.patches import Patch, PathPatch
from shapely.ops import unary_union
from matplotlib.path import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from parameter import ShapeParameters
from config import Config
from utils import ShapeUtils


PointType: TypeAlias = Tuple[float, float]

# ---------------------------
# 单形状变体处理器
# ---------------------------
class SingleShapeVariants:
    """单形状装饰、掩码、变形等变体处理"""
    @staticmethod
    def _ray_segment_intersection(ray_origin: np.ndarray, ray_dir: np.ndarray,
                                  p1: np.ndarray, p2: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
        """
        Solve ray_origin + s*ray_dir = p1 + t*(p2-p1)
        Return (s, point) with s >= 0 and t in [0,1], otherwise None.
        """
        v = p2 - p1  # edge vector
        A = np.column_stack((v, -ray_dir))  # v * t + (-ray_dir) * s = ray_origin - p1
        b = ray_origin - p1
        det = np.linalg.det(A)
        # if nearly parallel, skip
        if abs(det) < 1e-10:
            return None
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        t_param, s_param = sol[0], sol[1]
        if -1e-9 <= t_param <= 1 + 1e-9 and s_param >= -1e-9:
            inter = p1 + t_param * v
            return float(s_param), inter
        return None

    @staticmethod
    def _point_on_boundary(ax: plt.Axes, shape: Patch, angle: float) -> PointType:
        """
        返回沿 angle (radians, 相对于数据坐标轴，0 在 x 正方向) 从形状中心射出的第一处与边界相交的点。
        对于圆、椭圆做解析解；对于多边形/任意 PathPatch 做射线-边段相交的精确求解（transform 到数据坐标系后）。
        """
        cx, cy = ShapeUtils.get_center(shape, ax)
        if isinstance(shape, Wedge):
            cx, cy = shape.center  # Wedge 自带 center 属性 (cx, cy)
        origin = np.array([cx, cy], dtype=np.float64)
        dir_vec = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

        # --- Circle-like (解析)
        if isinstance(shape, Circle) or isinstance(shape, RegularPolygon):
            # 获取中心
            cx, cy = ShapeUtils.get_center(shape, ax)
            origin = np.array([cx, cy], dtype=np.float64)
            # 获取方向向量
            dir_vec = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

            if isinstance(shape, Circle):
                r = getattr(shape, "radius", None)
                if r is None:
                    r = getattr(shape, "get_radius", lambda: 0.0)()
                return (cx + r * dir_vec[0], cy + r * dir_vec[1])
            else:  # RegularPolygon
                # 获取顶点在 data 坐标系下（包含旋转和 scale）
                verts = shape.get_path().transformed(shape.get_patch_transform()).vertices
                # 射线-多边形交点
                ray_origin = origin
                ray_dir = dir_vec
                intersections = []
                for i in range(len(verts)):
                    p1 = verts[i].astype(np.float64)
                    p2 = verts[(i + 1) % len(verts)].astype(np.float64)
                    res = SingleShapeVariants._ray_segment_intersection(ray_origin, ray_dir, p1, p2)
                    if res is not None:
                        s_val, inter_pt = res
                        intersections.append((s_val, inter_pt))
                if intersections:
                    intersections.sort(key=lambda x: x[0])
                    pt = intersections[0][1]
                    return (float(pt[0]), float(pt[1]))
                return (cx, cy)

        # --- Wedge (扇形) - 若角度落在扇形弧段内，则交点是 r*dir；否则使用边界 path 求交（下面的通用分支会处理）
        if isinstance(shape, Wedge):
            # 注意 theta1/theta2 单位为度，theta1 -> theta2 的方向（matplotlib 默认 anti-clockwise?）
            theta1, theta2 = float(shape.theta1), float(shape.theta2)
            # 将 angle 转为度并规范化到 [0,360)
            ang_deg = (math.degrees(angle)) % 360.0
            # 统一比较（处理跨 360 的情况）
            def angle_in_arc(a, a1, a2):
                if a1 <= a2:
                    return a1 - 1e-9 <= a <= a2 + 1e-9
                else:
                    # 跨越 360 场景
                    return a >= a1 - 1e-9 or a <= a2 + 1e-9
            if angle_in_arc(ang_deg, theta1 % 360.0, theta2 % 360.0):
                r = getattr(shape, "r", 0.0)
                return (cx + r * dir_vec[0], cy + r * dir_vec[1])
            # 否则让下面的通用路径/边段求交来处理（会找到弧的离散化或直边的交点）

        # --- Ellipse (解析解)
        if isinstance(shape, Ellipse):
            a = float(shape.width) / 2.0
            b = float(shape.height) / 2.0
            phi = math.radians(getattr(shape, "angle", 0.0))  # 旋转角度（度 -> 弧度）
            # 将方向向量旋转到椭圆主轴坐标系（等于以 -phi 旋转 dir_vec）
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            dx = dir_vec[0]
            dy = dir_vec[1]
            dxp = cos_phi * dx + sin_phi * dy
            dyp = -sin_phi * dx + cos_phi * dy
            denom = (dxp * dxp) / (a * a) + (dyp * dyp) / (b * b)
            if denom <= 0:
                # 退化情况，返回中心
                return (cx, cy)
            s = math.sqrt(1.0 / denom)
            return (cx + s * dir_vec[0], cy + s * dir_vec[1])

        # --- 任意 Path / 多边形（通用求交）
        # 尝试先获取顶点：优先使用 get_xy()（Polygon），否则使用 get_path() + transform
        verts = None
        try:
            if hasattr(shape, "get_xy"):
                verts = np.asarray(shape.get_xy(), dtype=np.float64)
                # 有些实现会包含最后一个重复闭合点，去掉最后一个
                if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
                    verts = verts[:-1]
            else:
                # use path + transform to get coordinates in data space
                path = shape.get_path()
                transform = shape.get_transform()
                raw = np.asarray(path.vertices, dtype=np.float64)
                verts = transform.transform(raw)
                if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
                    verts = verts[:-1]
        except Exception as e:
            logging.debug("无法取得 shape 顶点，返回中心: %s", e)
            return (cx, cy)

        if verts is None or len(verts) < 2:
            return (cx, cy)

        # 做射线-每条边的交点求解，取最小的 s >= 0
        ray_origin = origin
        ray_dir = dir_vec
        intersections = []
        for i in range(len(verts)):
            p1 = verts[i].astype(np.float64)
            p2 = verts[(i + 1) % len(verts)].astype(np.float64)
            res = SingleShapeVariants._ray_segment_intersection(ray_origin, ray_dir, p1, p2)
            if res is not None:
                s_val, inter_pt = res
                intersections.append((s_val, inter_pt))

        if intersections:
            intersections.sort(key=lambda x: x[0])
            pt = intersections[0][1]
            return (float(pt[0]), float(pt[1]))
        # 最后保底（没有交点）
        return (cx, cy)

    # ---------- helper: wedge angle utilities ----------
    def _wedge_angle_range(shape):
        """
        如果 shape 是 Wedge，返回 (a1, a2) 两个角（弧度），表示从 a1 到 a2 按 增加方向（matplotlib 默认）
        的区间；否则返回 full circle (0, 2*pi).
        注意：theta1/theta2 在 matplotlib Wedge 中是以度为单位，且可能 a2 < a1（表示跨越 360）。
        返回的 a1,a2 都规范化到 [0, 2*pi).
        """
        if not isinstance(shape, Wedge):
            return 0.0, 2 * math.pi

        t1 = float(shape.theta1) % 360.0
        t2 = float(shape.theta2) % 360.0
        a1 = math.radians(t1)
        a2 = math.radians(t2)
        # 保证在 [0,2pi)
        a1 = a1 % (2 * math.pi)
        a2 = a2 % (2 * math.pi)
        return a1, a2

    def _angle_in_arc(angle, a1, a2, eps=1e-9):
        """判断 angle (radians, 已 normalize 到 [0,2pi)) 是否在从 a1 到 a2 的弧段上（包含边界），支持跨越 2π 的情况。"""
        angle = angle % (2 * math.pi)
        a1 = a1 % (2 * math.pi)
        a2 = a2 % (2 * math.pi)
        if a1 <= a2:
            return (a1 - eps) <= angle <= (a2 + eps)
        else:
            # 跨越 2π, 区间是 [a1, 2π) U [0, a2]
            return angle >= (a1 - eps) or angle <= (a2 + eps)

    def _sample_angle_in_arc(a1, a2):
        """
        在弧段 [a1 -> a2] 上均匀采样一个角（弧度），正确处理 a2 < a1 的跨越情况。
        """
        a1 = a1 % (2 * math.pi)
        a2 = a2 % (2 * math.pi)
        if a1 <= a2:
            return a1 + random.random() * (a2 - a1)
        else:
            # 跨越 2pi: arc length = (2pi - a1) + a2
            total = (2 * math.pi - a1) + a2
            r = random.random() * total
            if r <= (2 * math.pi - a1):
                return a1 + r
            else:
                return (r - (2 * math.pi - a1))  # 在 [0, a2]

    def add_internal_decoration(ax: mpl.axes.Axes, shape: Patch, shape_params: 'ShapeParameters',
                            style: str = "random", n: Optional[int] = None) -> None:
        """
        在 shape 内部添加装饰线/图案，并把生成的 artists 存入 shape_params.decoration_artists 列表（如果提供）。
        确保所有坐标/计算均基于 ax 的 data 坐标系。
        """

        # 初始化装饰 artists 存储
        if not hasattr(shape_params, "decoration_artists") or shape_params.decoration_artists is None:
            try:
                shape_params.decoration_artists = []
            except Exception:
                # 如果 shape_params 不是简单对象（极少见），忽略存储但继续绘制
                shape_params.decoration_artists = None

        center = ShapeUtils.get_center(shape, ax)

        if n is None and not isinstance(shape, Wedge):
            n = random.randint(1, 6)
        elif isinstance(shape, Wedge):
            n = random.randint(1, ((shape.theta2 - shape.theta1 + 360) % 360) // 45 + 1)
        shape_params.has_decoration = True
        shape_params.decoration_style = style

        # helper: call is_point_inside_shape with/without ax depending on signature
        def _is_inside(shp, pt):
            try:
                # prefer new signature with ax
                return ShapeUtils.is_point_inside_shape(shp, ax, pt)
            except TypeError:
                try:
                    return ShapeUtils.is_point_inside_shape(shp, pt)
                except Exception:
                    return False
            except Exception:
                return False
        # ---------- radial style ----------
        if style == "radial":
            a1, a2 = SingleShapeVariants._wedge_angle_range(shape)
            # 等分弧段（如果是 full circle, 就是 0..2pi）
            # 使用 endpoint=False 保持和原来 np.linspace 行为一致
            if a1 <= a2:
                angles = np.linspace(a1, a2, n, endpoint=False)
            else:
                # 跨越 2pi: 构建在 [a1, 2pi) + [0, a2)
                # 为简单起见，先生成 n 均匀参数 t in [0,1) 再映射到弧段
                t = np.linspace(0.0, 1.0, n, endpoint=False)
                total = (2 * math.pi - a1) + a2
                angles = (a1 + t * total) % (2 * math.pi)

            for angle in angles:
                end = SingleShapeVariants._point_on_boundary(ax, shape, angle)
                ln_objs = ax.plot([center[0], end[0]], [center[1], end[1]],
                                linewidth=random.uniform(0.6, 1.4),
                                linestyle=random.choice(Config.LINE_STYLES),
                                alpha=0.9)
                if ln_objs:
                    ln = ln_objs[0]
                    if shape_params.decoration_artists is not None:
                        shape_params.decoration_artists.append(ln)

        # ---------- grid style ----------
        elif style == "grid":
            try:
                bb = ShapeUtils.get_bbox(shape, ax)  # data-space Bbox
                x0, y0, x1, y1 = float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1)
            except Exception:
                # fallback small bbox around center
                x0, y0 = center[0] - 0.5, center[1] - 0.5
                x1, y1 = center[0] + 0.5, center[1] + 0.5
            # horizontal stripes
            for i in range(1, n + 1):
                segs = random.randint(2, 6)
                y_fixed = y0 + (y1 - y0) * i / n
                x_samples = np.linspace(x0, x1, 200)
                inside_x = []
                for x in x_samples:
                    if _is_inside(shape, (x, y_fixed)):
                        inside_x.append(x)
                if inside_x:
                    ln_objs = ax.plot([min(inside_x), max(inside_x)], [y_fixed, y_fixed],
                                    linewidth=1.2, alpha=0.8,
                                    linestyle=random.choice(Config.LINE_STYLES))
                    if ln_objs and shape_params.decoration_artists is not None:
                        shape_params.decoration_artists.append(ln_objs[0])

            # vertical stripes
            for i in range(1, n):
                segs = random.randint(2, 6)
                x_fixed = x0 + (x1 - x0) * i / n
                y_samples = np.linspace(y0, y1, 200)
                inside_y = []
                for y in y_samples:
                    if _is_inside(shape, (x_fixed, y)):
                        inside_y.append(y)
                if inside_y:
                    ln_objs = ax.plot([x_fixed, x_fixed], [min(inside_y), max(inside_y)],
                                    linewidth=1.2, alpha=0.8,
                                    linestyle=random.choice(Config.LINE_STYLES))
                    if ln_objs and shape_params.decoration_artists is not None:
                        shape_params.decoration_artists.append(ln_objs[0])

        # ---------- polygon style ----------
        elif style == "polygon":
            m = random.randint(3, 8)
            a1, a2 = SingleShapeVariants._wedge_angle_range(shape)
            if isinstance(shape, Wedge):
                # 在扇形弧段内随机采样 m 个角并排序（保证顺序环绕）
                angles = np.array([SingleShapeVariants._sample_angle_in_arc(a1, a2) for _ in range(m)])
                angles = np.sort(angles)
            else:
                # 非 Wedge: 在整圈内随机 m 个角并排序
                angles = np.sort(np.random.uniform(0, 2 * math.pi, m))

            # ensure we call point_on_boundary with ax
            boundary_points = [SingleShapeVariants._point_on_boundary(ax, shape, angle) for angle in angles]
            # close polygon
            x_coords = [p[0] for p in boundary_points] + [boundary_points[0][0]]
            y_coords = [p[1] for p in boundary_points] + [boundary_points[0][1]]
            ln_objs = ax.plot(x_coords, y_coords,
                            linewidth=random.uniform(0.8, 1.4),
                            linestyle=random.choice(Config.LINE_STYLES),
                            alpha=0.9,
                            color=random.choice(['black', 'gray', 'darkgray']))
            if ln_objs and shape_params.decoration_artists is not None:
                shape_params.decoration_artists.append(ln_objs[0])

        # ---------- random chords / connections ----------
        else:
            a1, a2 = SingleShapeVariants._wedge_angle_range(shape)
            for _ in range(n):
                if isinstance(shape, Wedge):
                    aa1 = SingleShapeVariants._sample_angle_in_arc(a1, a2)
                    aa2 = SingleShapeVariants._sample_angle_in_arc(a1, a2)
                else:
                    aa1 = random.uniform(0, 2 * math.pi)
                    aa2 = random.uniform(0, 2 * math.pi)
                p1 = SingleShapeVariants._point_on_boundary(ax, shape, aa1)
                p2 = SingleShapeVariants._point_on_boundary(ax, shape, aa2)
                ln_objs = ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                linewidth=random.uniform(0.8, 1.2),
                                alpha=0.9)
                if ln_objs and shape_params.decoration_artists is not None:
                    shape_params.decoration_artists.append(ln_objs[0])

        # 最后：确保 shape 已被添加到 axes（避免重复添加）
        try:
            # check if same object is already in axes' patches
            already = False
            for p in ax.patches:
                if p is shape:
                    already = True
                    break
            if not already:
                ax.add_patch(shape)
        except Exception:
            # 如果无法检查或添加，忽略（绘制可能已完成）
            try:
                ax.add_patch(shape)
            except Exception:
                pass

        # （不在此处触发全局 redraw，让调用方决定何时刷新）
        return

    def apply_mask(ax: mpl.axes.Axes, base_shape: Patch, shape_params: Dict,
                mask_type: str = "random", fill_color: str = "white") -> None:
        """
        为形状添加掩码（利用其他形状遮盖），支持两种模式：
        - "cut"：被遮盖部分直接移除（原形状 - 遮盖形状），填充和边线都设为背景色（fill_color）
        - "replace_boundary"：填充设为背景色；原始边界中落在 mask 内的线段绘制为黑色，其余边段保持原边框色

        注意：该函数会移除 ax 中的 base_shape（如果存在），并替换为处理后的显示（patch 和 line）。
        """
        logging.debug("应用掩码：type=%s, 形状类型=%s", mask_type, type(base_shape))
        center = ShapeUtils.get_center(base_shape, ax)

        base_facecolor = base_shape.get_facecolor() if hasattr(base_shape, "get_facecolor") else fill_color
        base_edgecolor = base_shape.get_edgecolor() if hasattr(base_shape, "get_edgecolor") else None
        base_linewidth = base_shape.get_linewidth() if hasattr(base_shape, "get_linewidth") else 2.0

        # -------------------------- 辅助函数：形状与几何对象转换 --------------------------
        def base_shape_to_shapely(shape: Patch) -> Polygon:
            """将 matplotlib Patch 转为 shapely Polygon（用于几何运算）"""
            if isinstance(shape, mpl.patches.Circle):
                cx, cy = shape.center
                return Point(cx, cy).buffer(shape.radius)
            elif isinstance(shape, mpl.patches.Rectangle):
                # rectangle: get_xy() 返回左下角
                x, y = shape.get_xy()
                # 有些 Rectangle 存储属性名不同，兼容读取
                w = getattr(shape, "_width", getattr(shape, "get_width", lambda: shape.get_bbox().width)())
                h = getattr(shape, "_height", getattr(shape, "get_height", lambda: shape.get_bbox().height)())
                return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            elif isinstance(shape, mpl.patches.Polygon):
                return Polygon(shape.get_xy())
            elif isinstance(shape, mpl.patches.Ellipse):
                cx, cy = shape.center
                w_half, h_half = shape.width / 2.0, shape.height / 2.0
                angles = np.linspace(0, 2 * np.pi, 100)
                points = [(cx + w_half * np.cos(a), cy + h_half * np.sin(a)) for a in angles]
                return Polygon(points)
            else:
                # 默认使用边界框近似
                bbox = getattr(shape, "get_bbox", lambda: None)()
                if bbox is not None:
                    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                # 最后兜底：空的很小矩形
                return Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])

        def create_random_mask_shapes(base_geom: Polygon, n: int = 1) -> List[Dict]:
            """创建 1~n 个随机遮盖形状（圆形/矩形），确保在原形状内部"""
            masks = []
            min_x, min_y, max_x, max_y = base_geom.bounds
            for _ in range(n):
                # 随机位置（在 base 内）
                for _i in range(1000):
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(min_y, max_y)
                    if base_geom.contains(Point(x, y)):
                        break
                else:
                    # 若找不到合适点则退回中心
                    x, y = ( (min_x+max_x)/2.0, (min_y+max_y)/2.0 )

                base_size = min(max_x - min_x, max_y - min_y)
                mask_size = base_size * random.uniform(0.5, 1.2)

                if random.random() < 0.5:
                    # 圆形遮盖
                    radius = mask_size / 2.0
                    mask_geom = Point(x, y).buffer(radius)
                    masks.append({
                        "type": "circle",
                        "geom": mask_geom,
                        "params": {"center": (x, y), "radius": radius}
                    })
                else:
                    # 矩形遮盖
                    width = mask_size * random.uniform(0.8, 1.2)
                    height = mask_size * random.uniform(0.8, 1.2)
                    x0, y0 = x - width / 2.0, y - height / 2.0
                    mask_geom = Polygon([(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)])
                    masks.append({
                        "type": "rectangle",
                        "geom": mask_geom,
                        "params": {"xy": (x0, y0), "width": width, "height": height}
                    })
            return masks

        def linestrings_to_line2d(lines, linewidth: float = 1.0, color='black', zorder: int = 3) -> List[Line2D]:
            """将 LineString / MultiLineString / list 转为 matplotlib Line2D 列表"""
            out = []
            if lines is None:
                return out
            if isinstance(lines, LineString):
                lines = [lines]
            elif isinstance(lines, MultiLineString):
                lines = list(lines.geoms)
            elif isinstance(lines, (list, tuple)):
                lines = list(lines)
            else:
                # 其他类型返回空
                return out
            for ls in lines:
                try:
                    coords = list(ls.coords)
                except Exception:
                    continue
                if len(coords) < 2:
                    continue
                xs, ys = zip(*coords)
                ln = Line2D(xs, ys, linewidth=linewidth, color=color, solid_capstyle='butt', zorder=zorder)
                out.append(ln)
            return out

        # -------------------------- 核心逻辑：生成遮盖并运算 --------------------------
        # 1. 将原形状转换为 shapely 几何对象（用于布尔运算）
        base_geom = base_shape_to_shapely(base_shape)

        # 2. 创建遮盖形状（1~3 个，随机类型）
        n_masks = random.randint(1, 3)
        mask_list = create_random_mask_shapes(base_geom, n_masks)
        mask_geoms = [m["geom"] for m in mask_list]
        mask_union = unary_union(mask_geoms) if mask_geoms else Polygon()

        # 3. 随机模式决定具体的 mask_type（如果传 "random"）
        if mask_type == "random":
            mask_type = "cut" if random.random() < 0.5 else "replace_boundary"

        # 4. 计算差集（保留未被遮盖的部分）
        try:
            result_geom = base_geom.difference(mask_union)
        except Exception as e:
            logging.warning("shapely 差集异常，回退到原形状: %s", e)
            result_geom = base_geom

        # 如果 result_geom 是 Multi 或 GeometryCollection，选择合适处理（但边界段的判断基于原始 base_geom）
        # 这里保留所有 polygon 的填充，以防你希望显示多个碎片；如果你只想保留最大区域，可用前面的 max 逻辑。
        polys_for_fill = []
        if isinstance(result_geom, MultiPolygon):
            polys_for_fill = list(result_geom.geoms)
        elif isinstance(result_geom, GeometryCollection):
            # 从 collection 中抽取多边形
            polys_for_fill = [g for g in result_geom.geoms if isinstance(g, Polygon)]
        elif isinstance(result_geom, Polygon):
            polys_for_fill = [result_geom]
        else:
            polys_for_fill = []

        # -------------------------- 更新显示与参数记录（移除原形状） --------------------------
        if base_shape in ax.patches:
            try:
                base_shape.remove()
            except ValueError:
                pass

        # 绘制填充区域（使用填充色 fill_color，edgecolor 暂不绘制）
        for poly in polys_for_fill:
            if poly.is_empty:
                continue
            ext_coords = list(poly.exterior.coords)
            if len(ext_coords) < 3:
                continue
            codes = [Path.MOVETO] + [Path.LINETO] * (len(ext_coords) - 2) + [Path.CLOSEPOLY]
            patch = PathPatch(Path(ext_coords, codes),
                            facecolor=fill_color,    # 填充为背景色（掏空效果）
                            edgecolor='none',
                            linewidth=0,
                            zorder=1)
            ax.add_patch(patch)

        # -------------------------- 关键：计算并绘制原始边界的被遮盖段（黑色）与保留段（原边框色） --------------------------
        # 原始边界（LineString 或 MultiLineString）
        orig_boundary = base_geom.boundary
        mask_boundary = mask_union.boundary

        # 被遮盖的原始边界段：mask边界与 base_geom 的交集 -> 这些段要画黑色
        try:
            cut_boundary_segments = mask_boundary.intersection(base_geom)
        except Exception:
            cut_boundary_segments = None

        # 原始边界中未被遮盖的段 -> 保持原边框色
        try:
            kept_boundary_segments = orig_boundary.difference(mask_union)
        except Exception:
            kept_boundary_segments = orig_boundary

        # 可选：过滤掉非常短的线段（避免噪音）
        MIN_SEG_LENGTH = 1e-3  # 依据坐标尺度可放大，例如 0.5
        def filter_short_lines(ml):
            if ml is None:
                return None
            if isinstance(ml, LineString):
                return ml if ml.length >= MIN_SEG_LENGTH else None
            if isinstance(ml, MultiLineString):
                geoms = [g for g in ml.geoms if g.length >= MIN_SEG_LENGTH]
                if not geoms:
                    return None
                return MultiLineString(geoms)
            # list/iterable
            if isinstance(ml, (list, tuple)):
                geoms = [g for g in ml if isinstance(g, LineString) and g.length >= MIN_SEG_LENGTH]
                if not geoms:
                    return None
                return MultiLineString(geoms)
            return ml

        cut_boundary_segments = filter_short_lines(cut_boundary_segments)
        kept_boundary_segments = filter_short_lines(kept_boundary_segments)

        # 绘制规则：
        linewidth_draw = float(base_linewidth if base_linewidth is not None else 1.0)
        # kept 的颜色优先使用 base_edgecolor，否则退回到 fill_color（避免 'none' 导致不可见）
        keep_color = base_edgecolor if (base_edgecolor is not None and base_edgecolor != 'none') else fill_color

        # 模式 cut：全部线段与填充都用背景色（此时我们已用 fill_color 填充；这里将 kept 与 cut 都绘制为 fill_color）
        if mask_type == "cut":
            # 将所有原始边界段都绘制为背景色（使得边框看起来消失）
            all_lines = []
            if kept_boundary_segments is not None:
                all_lines.extend(linestrings_to_line2d(kept_boundary_segments, linewidth=linewidth_draw, color='black', zorder=2))
            for ln in all_lines:
                print(ln)
                ax.add_line(ln)
            # done for cut
            return

        # 模式 replace_boundary：填充为背景色；未被遮盖段绘制为 keep_color；被遮盖段绘制为黑色
        if mask_type == "replace_boundary":
            # 先绘制未被遮盖的边段（外轮廓的那些段）
            if kept_boundary_segments is not None:
                for ln in linestrings_to_line2d(kept_boundary_segments, linewidth=linewidth_draw, color='black', zorder=3):
                    ax.add_line(ln)
            # 再绘制被遮盖的边段（黑色）
            if cut_boundary_segments is not None:
                for ln in linestrings_to_line2d(cut_boundary_segments, linewidth=linewidth_draw, color='black', zorder=4):
                    ax.add_line(ln)
            return


    @staticmethod
    def deform_edge(shape: Patch) -> None:
        """变形多边形边缘（仅对Polygon有效）"""
        if not isinstance(shape, Polygon):
            logging.debug("变形边缘：非多边形，跳过")
            return

        try:
            verts = np.asarray(shape.get_xy()[:-1], dtype=np.float32).copy()  # 移除闭合点
            if len(verts) < 2:
                logging.debug("变形边缘：多边形顶点数不足，跳过")
                return
        except Exception as e:
            logging.debug("获取多边形顶点失败：%s", e)
            return

        new_verts = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            new_verts.append(tuple(p1))  # 保留原顶点

            # 计算中点并变形
            mid = 0.5 * (p1 + p2)
            if random.random() < 0.5:
                # 法向偏移
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]], dtype=np.float32)
                norm = np.linalg.norm(normal)
                if norm > 1e-8:
                    normal /= norm
                offset = random.uniform(-0.18, 0.18)
                deform = mid + normal * offset
            else:
                # 随机偏移
                deform = mid + np.array([
                    random.uniform(-0.12, 0.12),
                    random.uniform(-0.12, 0.12)
                ], dtype=np.float32)
            new_verts.append(tuple(deform))

        new_verts.append(tuple(new_verts[0]))  # 闭合多边形
        try:
            shape.set_xy(np.array(new_verts, dtype=np.float32))
            logging.debug("变形边缘：成功，新顶点数=%d", len(new_verts))
        except Exception as e:
            logging.debug("变形边缘：设置顶点失败：%s", e)