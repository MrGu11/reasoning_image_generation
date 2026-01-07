from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple, TypeAlias, Union, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
from matplotlib.patches import Patch, PathPatch
from shapely.geometry import Polygon

from parameter import ShapeParameters
from utils import ShapeUtils



# ---------------------------
# 多形状组合器
# ---------------------------
class MultiShapeCombinator:
    @staticmethod
    def nested(ax: plt.Axes, shapes: List[Patch], shape_params_list: List[ShapeParameters],
              same_center: bool = True, scale_factor: float = 0.4, min_size: float = 0.25) -> List[Dict]:
        """
        对多种类型的图形进行嵌套排列（从外到内缩放），支持 Circle,Ellipse,FancyBboxPatch,
        Polygon,Rectangle,RegularPolygon,Wedge。

        Args:
            ax: matplotlib Axes 对象。scaled 后的 patch 会被 add 到 ax 上。
            shapes: Patch 子类对象列表（外到内的顺序：第 0 个为最外层）。
            shape_params_list: 形状参数列表，与shapes一一对应
            same_center: 是否保持中心相同（如果 True，所有内层会以最外层中心为中心进行缩放和平移）。
            scale_factor: 每层相对于上一层的缩放比例（实现为 scale_factor**i）。
            min_size: 最小尺寸（对不同图形含义不同：矩形/椭圆为宽/高，圆形为半径等）。
        Returns:
            一个列表，每个元素为字典，包含变换信息
        """

        def _centroid_of_coords(coords: np.ndarray) -> Tuple[float, float]:
            # coords: (N,2)
            return float(coords[:, 0].mean()), float(coords[:, 1].mean())

        def _extract_info(p: Patch) -> Dict:
            info = {'type': type(p).__name__, 'orig_center': None, 'orig_size': None, 'extra': {}}
            if isinstance(p, Circle):
                cx, cy = p.center
                info['orig_center'] = (float(cx), float(cy))
                info['orig_size'] = float(p.radius)
            elif isinstance(p, Ellipse):
                cx, cy = p.center
                info['orig_center'] = (float(cx), float(cy))
                # Ellipse.width/height are full widths
                info['orig_size'] = (float(p.width), float(p.height))
                info['extra']['angle'] = getattr(p, 'angle', 0.0)
            elif isinstance(p, Rectangle):
                x, y = p.get_x(), p.get_y()
                w, h = p.get_width(), p.get_height()
                info['orig_center'] = (float(x + w / 2.0), float(y + h / 2.0))
                info['orig_size'] = (float(w), float(h))
            elif isinstance(p, FancyBboxPatch):
                # FancyBboxPatch has get_x/get_y/get_width/get_height like Rectangle
                x, y = p.get_x(), p.get_y()
                w, h = p.get_width(), p.get_height()
                info['orig_center'] = (float(x + w / 2.0), float(y + h / 2.0))
                info['orig_size'] = (float(w), float(h))
                # try to store boxstyle if present
                try:
                    info['extra']['boxstyle'] = p.get_boxstyle()
                except Exception:
                    info['extra']['boxstyle'] = None
            elif isinstance(p, Polygon):
                coords = np.asarray(p.get_xy())
                cx, cy = _centroid_of_coords(coords)
                bbox_w = float(coords[:, 0].max() - coords[:, 0].min())
                bbox_h = float(coords[:, 1].max() - coords[:, 1].min())
                info['orig_center'] = (cx, cy)
                info['orig_size'] = (bbox_w, bbox_h)
                info['extra']['coords'] = coords
            elif isinstance(p, RegularPolygon):
                # RegularPolygon stores .xy (center), .radius, .numVertices, .orientation
                center = getattr(p, 'xy', getattr(p, 'center', None))
                if center is None:
                    # fallback: try attribute 'center'
                    center = (0.0, 0.0)
                cx, cy = center
                info['orig_center'] = (float(cx), float(cy))
                info['orig_size'] = float(getattr(p, 'radius', getattr(p, 'r', 0.0)))
                info['extra']['numVertices'] = getattr(p, 'numVertices', getattr(p, 'numsides', None))
                info['extra']['orientation'] = getattr(p, 'orientation', 0.0)
            elif isinstance(p, Wedge):
                cx, cy = p.center
                info['orig_center'] = (float(cx), float(cy))
                info['orig_size'] = float(p.r)  # radius
                info['extra']['theta1'] = getattr(p, 'theta1', None)
                info['extra']['theta2'] = getattr(p, 'theta2', None)
            else:
                # fallback: try to use get_path or bbox
                try:
                    bbox = p.get_extents()
                    cx = (bbox.x0 + bbox.x1) / 2.0
                    cy = (bbox.y0 + bbox.y1) / 2.0
                    w = bbox.x1 - bbox.x0
                    h = bbox.y1 - bbox.y0
                    info['orig_center'] = (float(cx), float(cy))
                    info['orig_size'] = (float(w), float(h))
                except Exception:
                    info['orig_center'] = (0.0, 0.0)
                    info['orig_size'] = None
            return info

        def _clone_and_scale(info: Dict, orig_patch: Patch, scale: float, center_override: Union[None, Tuple[float, float]] = None) -> Patch:
            t = info['type']
            cx_o, cy_o = info['orig_center']
            cx_new, cy_new = (center_override if center_override is not None else (cx_o, cy_o))
            # helper to copy visual props
            face = getattr(orig_patch, 'get_facecolor', lambda: None)()
            edge = getattr(orig_patch, 'get_edgecolor', lambda: None)()
            lw = getattr(orig_patch, 'get_linewidth', lambda: None)()

            if t == 'Circle':
                r_o = float(info['orig_size'])
                r_new = max(r_o * scale, float(min_size))
                p_new = Circle((cx_new, cy_new), r_new,
                            facecolor=face, edgecolor=edge, linewidth=lw)
            elif t == 'Ellipse':
                w_o, h_o = info['orig_size']
                w_new = max(w_o * scale, float(min_size))
                h_new = max(h_o * scale, float(min_size))
                angle = info['extra'].get('angle', 0.0)
                p_new = Ellipse((cx_new, cy_new), w_new, h_new, angle=angle,
                                facecolor=face, edgecolor=edge, linewidth=lw)
            elif t == 'Rectangle':
                w_o, h_o = info['orig_size']
                w_new = max(w_o * scale, float(min_size))
                h_new = max(h_o * scale, float(min_size))
                x_new = cx_new - w_new / 2.0
                y_new = cy_new - h_new / 2.0
                p_new = Rectangle((x_new, y_new), w_new, h_new,
                                facecolor=face, edgecolor=edge, linewidth=lw)
            elif t == 'FancyBboxPatch':
                w_o, h_o = info['orig_size']
                w_new = max(w_o * scale, float(min_size))
                h_new = max(h_o * scale, float(min_size))
                x_new = cx_new - w_new / 2.0
                y_new = cy_new - h_new / 2.0
                boxstyle = info['extra'].get('boxstyle', None)
                if boxstyle is not None:
                    try:
                        p_new = FancyBboxPatch((x_new, y_new), w_new, h_new,
                                            boxstyle=boxstyle,
                                            facecolor=face, edgecolor=edge, linewidth=lw)
                    except Exception:
                        # fallback to Rectangle if boxstyle reconstruction fails
                        p_new = Rectangle((x_new, y_new), w_new, h_new,
                                        facecolor=face, edgecolor=edge, linewidth=lw)
                else:
                    p_new = Rectangle((x_new, y_new), w_new, h_new,
                                    facecolor=face, edgecolor=edge, linewidth=lw)
            elif t == 'Polygon':
                coords = np.asarray(info['extra']['coords'])
                cx0, cy0 = info['orig_center']
                # scale coordinates about their centroid (or about provided center_override)
                center_for_scale = np.array([cx_new, cy_new])
                scaled = (coords - np.array([cx0, cy0])) * scale + center_for_scale
                p_new = Polygon(scaled, closed=True, facecolor=face, edgecolor=edge, linewidth=lw)
            elif t == 'RegularPolygon':
                r_o = float(info['orig_size'])
                r_new = max(r_o * scale, float(min_size))
                nverts = info['extra'].get('numVertices', None)
                orient = info['extra'].get('orientation', 0.0)
                if nverts is None:
                    # try to read attribute
                    nverts = getattr(orig_patch, 'numVertices', getattr(orig_patch, 'numsides', 3))
                p_new = RegularPolygon((cx_new, cy_new), int(nverts), radius=r_new, orientation=orient,
                                    facecolor=face, edgecolor=edge, linewidth=lw)
            elif t == 'Wedge':
                r_o = float(info['orig_size'])
                r_new = max(r_o * scale, float(min_size))
                theta1 = info['extra'].get('theta1', getattr(orig_patch, 'theta1', 0.0))
                theta2 = info['extra'].get('theta2', getattr(orig_patch, 'theta2', 360.0))
                p_new = Wedge((cx_new, cy_new), r_new, theta1, theta2,
                            facecolor=face, edgecolor=edge, linewidth=lw)
            else:
                # fallback: try bounding box clone as Rectangle
                orig_size = info.get('orig_size')
                if orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                    w_o, h_o = orig_size
                    w_new = max(w_o * scale, float(min_size))
                    h_new = max(h_o * scale, float(min_size))
                    x_new = cx_new - w_new / 2.0
                    y_new = cy_new - h_new / 2.0
                    p_new = Rectangle((x_new, y_new), w_new, h_new,
                                    facecolor=face, edgecolor=edge, linewidth=lw)
                else:
                    # last resort: return original (no scaling)
                    p_new = orig_patch
            return p_new

        if not shapes or len(shapes) != len(shape_params_list):
            return []

        # extract info for each original shape
        infos = [_extract_info(p) for p in shapes]
        # outer center (for same_center behavior)
        outer_center = infos[0]['orig_center']

        results = []
        for i, (orig_patch, info, params) in enumerate(zip(shapes, infos, shape_params_list)):
            scale = float(scale_factor ** i) if i >= 1 else 1.0
            # decide center: if same_center -> outer_center for all; else -> original center
            center_use = outer_center if same_center else info['orig_center']

            # create scaled patch
            p_new = _clone_and_scale(info, orig_patch, scale=scale, center_override=center_use)

            # add to axes
            try:
                ax.add_patch(p_new)
            except Exception:
                pass  # safety: some fallback patch might fail to add, ignore

            # 更新参数记录
            new_center = ShapeUtils.get_center(p_new, ax)
            new_bbox = ShapeUtils.get_bbox(p_new, ax)
            
            # 更新尺寸信息
            if isinstance(p_new, (Circle, Wedge)):
                new_size = float(getattr(p_new, "radius", getattr(p_new, "r", 0.0)))
            elif isinstance(p_new, Ellipse):
                new_size = (float(p_new.width), float(p_new.height))
            elif isinstance(p_new, (Rectangle, FancyBboxPatch)):
                new_size = (float(p_new.get_width()), float(p_new.get_height()))
            elif isinstance(p_new, RegularPolygon):
                new_size = float(getattr(p_new, "radius", getattr(p_new, "r", 0.0)))
            elif isinstance(p_new, Polygon):
                x0, y0, x1, y1 = new_bbox
                new_size = (x1 - x0, y1 - y0)
            else:
                new_size = params.size
                
            params.center = new_center
            params.bbox = new_bbox
            params.size = new_size
            params.extra_params["scale_factor"] = scale

            # 替换原形状
            shapes[i] = p_new

            entry = {
                'type': info['type'],
                'scale': scale,
                'new_center': new_center
            }
            results.append(entry)

        # optionally autoscale view so all patches visible
        # try:
        #     ax.autoscale_view()
        # except Exception:
        #     pass

        return results
    
    @staticmethod
    def _adaptive_eps(poly1: np.ndarray, poly2: np.ndarray = None) -> float:
        """基于输入多边形尺度自适应的数值容差（比固定的 1e-12 更稳健）。
        取两多边形包围盒对角线长度的一个很小的比例作为 eps。
        """
        def bbox_diag(poly: np.ndarray) -> float:
            if poly is None or poly.size == 0:
                return 0.0
            mn = poly.min(axis=0)
            mx = poly.max(axis=0)
            return float(np.linalg.norm(mx - mn))

        d1 = bbox_diag(poly1)
        d2 = bbox_diag(poly2) if poly2 is not None else 0.0
        base = max(d1, d2, 1.0)
        # 1e-9 相对于尺度的安全容差，且不小于 1e-12
        return max(1e-12, base * 1e-9)

    @staticmethod
    def _convex_hull(points: np.ndarray) -> np.ndarray:
        """Andrew 算法凸包实现；保留顺序并只去掉真正重复的点。
        修改点：保留与边共线的点（避免因为去掉共线点而产生微小间隙）。
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) <= 1:
            return np.unique(pts, axis=0)

        pts_sorted = np.unique(pts, axis=0)
        pts_sorted = pts_sorted[np.lexsort((pts_sorted[:, 1], pts_sorted[:, 0]))]

        def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        eps = 1e-9
        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and cross(np.array(lower[-2]), np.array(lower[-1]), p) < -eps:
                lower.pop()
            lower.append(tuple(p))

        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and cross(np.array(upper[-2]), np.array(upper[-1]), p) < -eps:
                upper.pop()
            upper.append(tuple(p))

        hull_tuples = lower[:-1] + upper[:-1]
        out = []
        seen_prev = None
        for t in hull_tuples:
            if seen_prev is None or (abs(t[0] - seen_prev[0]) > eps or abs(t[1] - seen_prev[1]) > eps):
                out.append(t)
            seen_prev = t
        hull = np.asarray(out, dtype=np.float64)
        if hull.shape[0] == 0:
            return pts_sorted
        return hull

    @staticmethod
    def _centroid(verts: np.ndarray) -> np.ndarray:
        """顶点平均作为质心（简化）"""
        if verts.size == 0:
            return np.array([0.0, 0.0], dtype=np.float64)
        return verts.mean(axis=0)

    @staticmethod
    def _unit_vector(angle: float) -> np.ndarray:
        return np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

    @staticmethod
    def _project_polygon(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        projs = poly.dot(axis)
        return float(projs.min()), float(projs.max())

    @staticmethod
    def _support_max(poly: np.ndarray, u: np.ndarray) -> float:
        return float(np.max(poly.dot(u)))

    @staticmethod
    def _support_min(poly: np.ndarray, u: np.ndarray) -> float:
        return float(np.min(poly.dot(u)))

    @staticmethod
    def _min_separation_and_axis(poly1: np.ndarray, poly2: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
        """
        使用 SAT 计算 poly1 与 poly2 在所有轴上的最大分离（>=0 表示存在间隙）。
        返回 (separation, axis)，axis 为单位向量，方向是把 poly1 推向 poly2 的方向。
        separation 是沿该轴需要推进的最小距离以实现接触。
        """
        adaptive = MultiShapeCombinator._adaptive_eps(poly1, poly2)

        def axes_from(poly: np.ndarray) -> List[np.ndarray]:
            axes = []
            n = len(poly)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                edge = b - a
                axis = np.array([-edge[1], edge[0]], dtype=np.float64)
                norm = np.linalg.norm(axis)
                if norm > adaptive:
                    axes.append(axis / norm)
            unique_axes = []
            for ax in axes:
                if not any(np.allclose(ax, ux, atol=max(1e-8, adaptive)) or np.allclose(ax, -ux, atol=max(1e-8, adaptive)) for ux in unique_axes):
                    unique_axes.append(ax)
            return unique_axes

        axes = axes_from(poly1) + axes_from(poly2)
        if len(axes) == 0:
            return 0.0, np.array([0.0, 0.0], dtype=np.float64)

        best_sep = -np.inf
        best_axis = None

        for axis in axes:
            p1_min, p1_max = MultiShapeCombinator._project_polygon(poly1, axis)
            p2_min, p2_max = MultiShapeCombinator._project_polygon(poly2, axis)
            d1 = p2_min - p1_max
            d2 = p1_min - p2_max
            sep = max(d1, d2)
            # axis sign: 希望 axis 指向把 poly1 推向 poly2 的方向
            chosen_axis = axis.copy() if d1 >= d2 else -axis.copy()
            if sep > best_sep:
                best_sep = sep
                best_axis = chosen_axis

        if best_axis is None:
            return 0.0, np.array([0.0, 0.0], dtype=np.float64)
        return float(max(0.0, best_sep)), best_axis

    @staticmethod
    def _iterative_snap_and_apply(curr_hull: np.ndarray,
                                  candidate_translation: np.ndarray,
                                  placed_hulls: List[np.ndarray],
                                  *,
                                  gap_tol: float = 1e-8,
                                  max_outer_iters: int = 8,
                                  bs_iters: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """
        迭代二分贴合：
        - 对每个已放置 hull 计算 (sep, axis)。
        - 若 sep 在可接受的小间隙阈值内（<= gap_tol），尝试沿 axis 推进，
          使用二分查找找到能推进的最大 fraction（允许接触但不允许穿透）。
        - 重复外部迭代直到无进展或达到上限。
        返回 (updated_curr_hull, updated_translation)
        修改点：
        * 使用自适应 eps
        * 二分检查时允许接触（allow_touching=True），以尽量消除微小间隙
        * tiny_backoff 基于尺度自适应，避免硬编码 1e-12
        """
        candidate_hull = curr_hull + candidate_translation
        translation = candidate_translation.copy()

        if not placed_hulls:
            return candidate_hull, translation

        for outer in range(max_outer_iters):
            progress = False
            # 计算分离
            seps = []
            for idx, ph in enumerate(placed_hulls):
                sep, axis = MultiShapeCombinator._min_separation_and_axis(candidate_hull, ph)
                seps.append((idx, sep, axis))

            max_sep = max((s for _i, s, _ax in seps), default=0.0)
            if max_sep <= gap_tol:
                break

            # 从大到小尝试
            seps_sorted = sorted(seps, key=lambda x: -x[1])
            for idx, sep, axis in seps_sorted:
                if sep <= 0.0:
                    continue
                if np.linalg.norm(axis) < 1e-16:
                    continue

                # 只处理合理范围内的小间隙（依据尺度自适应），避免把非常大间隙也当成数值问题
                adaptive = MultiShapeCombinator._adaptive_eps(candidate_hull, placed_hulls[idx])
                # 允许推进的上限：基于 hull 尺度的一个小比例或 gap_tol 的放大
                max_acceptable = max(gap_tol * 1e6, adaptive * 1000)
                if sep > max_acceptable:
                    continue

                desired_delta = axis * sep
                lo, hi = 0.0, 1.0
                safe_frac = 0.0
                others = [h for j, h in enumerate(placed_hulls) if j != idx]

                if not others:
                    # 如果没有其他障碍，直接把物体推进到接触
                    eps_margin = min(sep * 1e-6, max(gap_tol * 1e-3, adaptive * 1e-3))
                    apply_delta = desired_delta
                    candidate_hull = candidate_hull + apply_delta
                    translation = translation + apply_delta
                    progress = True
                    continue

                # 二分查找：注意这里允许接触（allow_touching=True），尽可能贴紧
                for _ in range(bs_iters):
                    mid = (lo + hi) / 2.0
                    trial_hull = candidate_hull + desired_delta * mid
                    collision = any(
                        MultiShapeCombinator._polygons_intersect_sat(trial_hull, other, eps=adaptive, allow_touching=True)
                        for other in others
                    )
                    if not collision:
                        safe_frac = mid
                        lo = mid
                    else:
                        hi = mid

                if safe_frac > 0.0:
                    # 使用自适应微退让，避免数值碰撞
                    tiny_backoff = max(adaptive * 1e-3, gap_tol * 1e-6)
                    frac_apply = max(0.0, safe_frac - tiny_backoff / max(abs(sep), adaptive))
                    delta_apply = desired_delta * frac_apply
                    candidate_hull = candidate_hull + delta_apply
                    translation = translation + delta_apply
                    progress = True

            if not progress:
                break

        return candidate_hull, translation

    @staticmethod
    def _get_vertices(patch: Patch, transform) -> np.ndarray:
        """获取 patch 在数据坐标下的顶点"""
        path = patch.get_path()
        return np.asarray(path.transformed(transform).vertices, dtype=np.float64)

    @staticmethod
    def _get_edges(vertices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        edges = []
        n = len(vertices)
        for i in range(n):
            a = vertices[i]
            b = vertices[(i + 1) % n]
            edges.append((a, b))
        return edges

    @staticmethod
    def _edges_parallel(edge1: Tuple[np.ndarray, np.ndarray],
                        edge2: Tuple[np.ndarray, np.ndarray],
                        eps: float = 1e-6) -> bool:
        vec1 = edge1[1] - edge1[0]
        vec2 = edge2[1] - edge2[0]
        cross = abs(vec1[0] * vec2[1] - vec1[1] * vec2[0])
        return cross < eps

    @staticmethod
    def _polygons_intersect_sat(poly1: np.ndarray, poly2: np.ndarray, eps: float = 1e-9, allow_touching: bool = True) -> bool:
        """使用 SAT 判断两个多边形是否相交（若 allow_touching 为 True，则“接触”被视为不相交）"""
        adaptive = max(eps, MultiShapeCombinator._adaptive_eps(poly1, poly2))

        def axes_from(poly: np.ndarray) -> List[np.ndarray]:
            axes = []
            n = len(poly)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                edge = b - a
                axis = np.array([-edge[1], edge[0]], dtype=np.float64)
                norm = np.linalg.norm(axis)
                if norm > adaptive:
                    axes.append(axis / norm)
            unique_axes = []
            for ax in axes:
                if not any(np.allclose(ax, ux, atol=max(1e-8, adaptive)) or np.allclose(ax, -ux, atol=max(1e-8, adaptive)) for ux in unique_axes):
                    unique_axes.append(ax)
            return unique_axes

        axes = axes_from(poly1) + axes_from(poly2)
        for axis in axes:
            min1, max1 = MultiShapeCombinator._project_polygon(poly1, axis)
            min2, max2 = MultiShapeCombinator._project_polygon(poly2, axis)
            if allow_touching:
                # 接触视为不相交（允许接触以消除微小间隙）
                if max1 <= min2 + adaptive or max2 <= min1 + adaptive:
                    return False
            else:
                # 更严格：接触也视为相交
                if max1 < min2 - adaptive or max2 < min1 - adaptive:
                    return False
        return True


    @staticmethod
    def adjacent(ax: plt.Axes,
                 shapes: List[Patch],
                 shape_params_list: List[ShapeParameters],
                 mode: str = "random",
                 adjacency_type: str = "auto",
                 *,
                 spacing: float = 0.0,
                 angle: float = 0.0,
                 sample_n: int = 180,
                 fit_margin: float = 0,
                 use_convex_hull: bool = False,
                 gap_tol: float = 1e-8) -> List[Tuple[float, float]]:
        """
        将 shapes 邻接放置到 ax 上并返回每个图形最终中心坐标列表。
        参数:
            use_convex_hull: True 使用凸包做碰撞近似（更快但对凹形会留缝）；False 使用原始顶点（更精确但慢）。
            gap_tol: iterative snap 的小间隙阈值（数据坐标）。
        """
        if len(shapes) != len(shape_params_list):
            raise ValueError("shapes和shape_params_list长度必须一致")

        n = len(shapes)
        if n == 0:
            return []

        original_transforms = []
        raw_verts = []
        hulls = []
        centroids = []
        vertices_list = []
        edges_list = []

        for p in shapes:
            if p.axes:
                p.remove()
            T_orig = p.get_patch_transform()
            original_transforms.append(T_orig)
            verts = MultiShapeCombinator._get_vertices(p, T_orig)
            raw_verts.append(verts)
            vertices_list.append(verts)
            edges_list.append(MultiShapeCombinator._get_edges(verts))
            if use_convex_hull:
                hull = MultiShapeCombinator._convex_hull(verts)
                hulls.append(hull if hull.shape[0] >= 2 else verts)
            else:
                hulls.append(verts)
            centroids.append(MultiShapeCombinator._centroid(verts))

        placed_centers: List[Tuple[float, float]] = []
        placed_hulls: List[np.ndarray] = []
        placed_vertices: List[np.ndarray] = []
        placed_edges: List[List[Tuple[np.ndarray, np.ndarray]]] = []

        # 放置第一个
        first_centroid = centroids[0]
        placed_centers.append((float(first_centroid[0]), float(first_centroid[1])))
        placed_hulls.append(hulls[0].copy())
        placed_vertices.append(vertices_list[0].copy())
        placed_edges.append(edges_list[0].copy())

        delta0 = np.array(placed_centers[0]) - first_centroid
        ShapeUtils.translate_shape(shapes[0], float(delta0[0]), float(delta0[1]), ax)
        ax.add_patch(shapes[0])

        two_pi = 2.0 * math.pi
        rng = random.Random(0)

        for i in range(1, n):
            curr_hull = hulls[i].copy()
            curr_centroid = centroids[i].copy()
            curr_vertices = vertices_list[i].copy()
            curr_edges = edges_list[i].copy()

            ref_idx = np.argmin([
                math.hypot(curr_centroid[0] - cx, curr_centroid[1] - cy)
                for cx, cy in placed_centers
            ])
            ref_hull = placed_hulls[ref_idx]
            ref_center = placed_centers[ref_idx]
            ref_vertices = placed_vertices[ref_idx]
            ref_edges = placed_edges[ref_idx]

            placed = False
            final_translation = np.array([0.0, 0.0], dtype=np.float64)

            # 点对点邻接
            if adjacency_type == "point":
                for ref_pt in ref_vertices:
                    for curr_pt in curr_vertices:
                        translation = ref_pt - curr_pt
                        candidate_hull = curr_hull + translation
                        collision = any(
                            MultiShapeCombinator._polygons_intersect_sat(candidate_hull, ph)
                            for ph in placed_hulls
                        )
                        if not collision:
                            final_translation = translation
                            placed = True
                            break
                    if placed:
                        break

            # 边对边邻接
            elif adjacency_type == "edge":
                for ref_edge in ref_edges:
                    ref_a, ref_b = ref_edge
                    ref_vec = ref_b - ref_a
                    ref_len = np.linalg.norm(ref_vec)
                    if ref_len < 1e-12:
                        continue
                    ref_unit = ref_vec / ref_len
                    ref_mid = (ref_a + ref_b) / 2
                    for curr_edge in curr_edges:
                        curr_a, curr_b = curr_edge
                        if not MultiShapeCombinator._edges_parallel(ref_edge, curr_edge):
                            continue
                        curr_vec = curr_b - curr_a
                        curr_mid = (curr_a + curr_b) / 2
                        norm_dir = np.array([-ref_unit[1], ref_unit[0]], dtype=np.float64)
                        base_trans = ref_mid - curr_mid + norm_dir * spacing
                        max_shift = max(ref_len, np.linalg.norm(curr_vec))
                        samples = 31
                        best = None
                        for s in np.linspace(-max_shift, max_shift, samples):
                            cand_trans = base_trans + ref_unit * s + norm_dir * fit_margin
                            candidate_hull = curr_hull + cand_trans
                            collision = any(
                                MultiShapeCombinator._polygons_intersect_sat(candidate_hull, ph, eps=1e-9, allow_touching=True)
                                for ph in placed_hulls
                            )
                            if not collision:
                                best = cand_trans
                                break
                        if best is not None:
                            final_translation = best
                            placed = True
                            break
                    if placed:
                        break

            # 自动模式：基于方向采样的 support 推动
            if not placed or adjacency_type == "auto":
                base_angles = np.linspace(angle, angle + two_pi, sample_n, endpoint=False)
                angles = list(base_angles)
                if mode == "random":
                    rng.shuffle(angles)
                scale = 1.0
                max_scale = 30
                for attempt in range(max_scale):
                    for th in angles:
                        u = MultiShapeCombinator._unit_vector(th)
                        s_ref = MultiShapeCombinator._support_max(ref_hull, u)
                        m_curr = MultiShapeCombinator._support_min(curr_hull, u)
                        t = (s_ref - m_curr) + spacing + fit_margin
                        translation = u * (t * scale)
                        candidate_hull = curr_hull + translation
                        if not any(
                            MultiShapeCombinator._polygons_intersect_sat(candidate_hull, ph)
                            for ph in placed_hulls
                        ):
                            final_translation = translation
                            placed = True
                            break
                    if placed:
                        break
                    scale *= 1.2

            # fallback 向外放置（粗略）
            if not placed:
                dir_vec = curr_centroid - np.array(ref_center, dtype=np.float64)
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-6:
                    dir_vec = np.array([1.0, 0.0], dtype=np.float64)
                else:
                    dir_vec /= norm
                s_ref = MultiShapeCombinator._support_max(ref_hull, dir_vec)
                m_curr = MultiShapeCombinator._support_min(curr_hull, dir_vec)
                t = (s_ref - m_curr) + spacing + fit_margin
                base_trans = dir_vec * t
                fa_scale = 1.0
                candidate_hull = curr_hull + base_trans * fa_scale
                fallback_attempts = 40
                while fallback_attempts > 0 and any(
                    MultiShapeCombinator._polygons_intersect_sat(candidate_hull, ph)
                    for ph in placed_hulls
                ):
                    fa_scale *= 1.25
                    candidate_hull = curr_hull + base_trans * fa_scale
                    fallback_attempts -= 1
                final_translation = base_trans * fa_scale
                placed = True

            # 迭代二分 snap（更稳健）
            candidate_hull, new_final_translation = MultiShapeCombinator._iterative_snap_and_apply(
                curr_hull=curr_hull,
                candidate_translation=final_translation,
                placed_hulls=placed_hulls,
                gap_tol=gap_tol,
                max_outer_iters=10,
                bs_iters=30
            )
            final_translation = new_final_translation

            # 更新放置信息
            new_centroid = curr_centroid + final_translation
            placed_centers.append((float(new_centroid[0]), float(new_centroid[1])))
            placed_hulls.append(curr_hull + final_translation)
            placed_vertices.append(curr_vertices + final_translation)
            placed_edges.append([(a + final_translation, b + final_translation) for a, b in curr_edges])

            # 应用平移并添加到轴
            ShapeUtils.translate_shape(shapes[i], float(final_translation[0]), float(final_translation[1]), ax)
            ax.add_patch(shapes[i])

        # 刷新画布（如在交互环境）
        try:
            ax.figure.canvas.draw_idle()
        except Exception:
            pass

        return placed_centers

    @staticmethod
    def _get_bbox(verts: np.ndarray) -> np.ndarray:
        """计算图形的边界框 [min_x, min_y, max_x, max_y]"""
        if verts.size == 0:
            # 处理空顶点的边缘情况
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        min_x = verts[:, 0].min()
        min_y = verts[:, 1].min()
        max_x = verts[:, 0].max()
        max_y = verts[:, 1].max()
        return np.array([min_x, min_y, max_x, max_y], dtype=np.float64)

    @staticmethod
    def intersecting(ax: mpl.axes.Axes, 
                    shapes: List[Patch], 
                    shape_params_list: List[ShapeParameters], 
                    overlap_style: str = "random") -> List[Tuple[float, float]]:
        """
        将图形以**实质性相交**方式放置（确保重叠区域，非仅接触）。
        参数:
            ax: 目标坐标轴
            shapes: 图形对象列表
            shape_params_list: 图形参数列表（与shapes一一对应）
            overlap_style: 相交模式 
                - "random": 随机与已放置图形相交，重叠程度随机
                - "center": 以已放置图形中心为基准，强制中心区域重叠
        返回:
            每个图形最终的中心坐标列表
        """
        if len(shapes) != len(shape_params_list):
            raise ValueError("shapes与shape_params_list长度必须一致")
        
        n = len(shapes)
        if n == 0:
            return []
        
        # 初始化：获取图形原始属性（复用已有逻辑，确保与邻接逻辑一致）
        original_transforms = []
        raw_verts = []
        hulls = []  # 凸包用于快速相交检测
        centroids = []  # 质心用于定位参考点
        bboxes = []  # 边界框用于控制相交范围
        
        for p in shapes:
            if p.axes:
                p.remove()  # 清除已有轴关联
            T_orig = p.get_patch_transform()
            original_transforms.append(T_orig)
            # 获取顶点（数据坐标）
            verts = MultiShapeCombinator._get_vertices(p, T_orig)
            raw_verts.append(verts)
            # 计算凸包（加速相交检测）
            hull = MultiShapeCombinator._convex_hull(verts)
            hulls.append(hull if hull.shape[0] >= 2 else verts)
            # 计算质心和边界框
            centroids.append(MultiShapeCombinator._centroid(verts))
            bboxes.append(MultiShapeCombinator._get_bbox(verts))  # 边界框：[min_x, min_y, max_x, max_y]
        
        # 存储已放置图形的状态
        placed_centers: List[Tuple[float, float]] = []
        placed_hulls: List[np.ndarray] = []
        placed_bboxes: List[np.ndarray] = []  # 已放置图形的边界框
        
        # 放置第一个图形（基准图形）
        first_centroid = centroids[0]
        first_bbox = bboxes[0]
        placed_centers.append((float(first_centroid[0]), float(first_centroid[1])))
        placed_hulls.append(hulls[0].copy())
        placed_bboxes.append(first_bbox)
        # 平移第一个图形到初始位置并添加到轴
        delta0 = np.array(placed_centers[0]) - first_centroid
        ShapeUtils.translate_shape(shapes[0], float(delta0[0]), float(delta0[1]), ax)
        ax.add_patch(shapes[0])
        
        rng = random.Random(42)  # 固定种子保证可复现
        two_pi = 2.0 * math.pi
        
        for i in range(1, n):
            curr_hull = hulls[i].copy()
            curr_centroid = centroids[i].copy()
            curr_bbox = bboxes[i]
            # 当前图形的尺度（用于控制相交程度）
            curr_size = np.linalg.norm([curr_bbox[2]-curr_bbox[0], curr_bbox[3]-curr_bbox[1]])  # 对角线长度
            
            # 选择参考图形（优先与最近的已放置图形相交）
            ref_idx = np.argmin([
                math.hypot(curr_centroid[0] - cx, curr_centroid[1] - cy)
                for cx, cy in placed_centers
            ])
            ref_hull = placed_hulls[ref_idx]
            ref_center = np.array(placed_centers[ref_idx], dtype=np.float64)
            ref_bbox = placed_bboxes[ref_idx]
            ref_size = np.linalg.norm([ref_bbox[2]-ref_bbox[0], ref_bbox[3]-ref_bbox[1]])
            
            final_translation: Optional[np.ndarray] = None
            
            if overlap_style == "random":
                # 随机相交模式：让新图形随机进入参考图形的边界框内，确保重叠
                # 1. 计算参考图形的内部区域（边界框向内收缩，确保进入后必相交）
                ref_inner_bbox = np.array([
                    ref_bbox[0] + ref_size * 0.2,  # 左边界右移20%
                    ref_bbox[1] + ref_size * 0.2,  # 下边界上移20%
                    ref_bbox[2] - ref_size * 0.2,  # 右边界左移20%
                    ref_bbox[3] - ref_size * 0.2   # 上边界下移20%
                ])
                # 若内部边界框无效（图形过小），直接用原始边界框
                if ref_inner_bbox[0] > ref_inner_bbox[2]:
                    ref_inner_bbox[0], ref_inner_bbox[2] = ref_bbox[0], ref_bbox[2]
                if ref_inner_bbox[1] > ref_inner_bbox[3]:
                    ref_inner_bbox[1], ref_inner_bbox[3] = ref_bbox[1], ref_bbox[3]
                
                # 2. 在参考图形内部区域随机采样目标中心，确保新图形与参考图形相交
                max_attempts = 50
                for _ in range(max_attempts):
                    # 在参考内部边界框内随机选点作为新图形的目标中心
                    target_cx = rng.uniform(ref_inner_bbox[0], ref_inner_bbox[2])
                    target_cy = rng.uniform(ref_inner_bbox[1], ref_inner_bbox[3])
                    target_centroid = np.array([target_cx, target_cy])
                    # 计算平移向量（从原始中心到目标中心）
                    translation = target_centroid - curr_centroid
                    candidate_hull = curr_hull + translation
                    
                    # 检查是否与参考图形实质性相交（非接触）
                    if MultiShapeCombinator._polygons_intersect_sat(
                        candidate_hull, ref_hull, allow_touching=False  # 严格排除仅接触
                    ):
                        final_translation = translation
                        break
                
                # 兜底：若随机采样失败，强制将新图形中心与参考中心重叠（确保相交）
                if final_translation is None:
                    final_translation = ref_center - curr_centroid  # 中心对齐（必然相交）
            
            elif overlap_style == "center":
                # 中心对齐相交：新图形中心与参考图形中心偏移一定比例，确保核心区域重叠
                offset_ratio = rng.uniform(0.3, 0.7)  # 偏移比例（0→完全重叠，1→边缘对齐）
                angle = rng.uniform(0, two_pi)  # 随机偏移方向
                # 偏移距离 = 参考图形尺度 * 偏移比例（确保进入参考图形内部）
                offset_dist = ref_size * offset_ratio * 0.5
                offset = np.array([
                    math.cos(angle) * offset_dist,
                    math.sin(angle) * offset_dist
                ])
                target_centroid = ref_center + offset  # 目标中心在参考中心附近
                final_translation = target_centroid - curr_centroid
                # 强制确保相交（若不相交则进一步缩小偏移）
                candidate_hull = curr_hull + final_translation
                if not MultiShapeCombinator._polygons_intersect_sat(
                    candidate_hull, ref_hull, allow_touching=False
                ):
                    final_translation = ref_center - curr_centroid  # 直接中心重叠
            
            else:
                raise ValueError(f"不支持的相交模式: {overlap_style}")
            
            # 应用平移并更新状态
            new_centroid = curr_centroid + final_translation
            placed_centers.append((float(new_centroid[0]), float(new_centroid[1])))
            placed_hulls.append(curr_hull + final_translation)
            # 更新边界框（平移后的边界框）
            new_bbox = curr_bbox + [final_translation[0], final_translation[1], 
                                   final_translation[0], final_translation[1]]
            placed_bboxes.append(new_bbox)
            
            # 平移图形并添加到轴
            ShapeUtils.translate_shape(shapes[i], 
                                      float(final_translation[0]), 
                                      float(final_translation[1]),
                                      ax)
            ax.add_patch(shapes[i])
        
        # 刷新画布
        try:
            ax.figure.canvas.draw_idle()
        except Exception:
            pass
        
        return placed_centers