from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import (
    Circle, Ellipse, FancyBboxPatch, Patch, Polygon, Rectangle, RegularPolygon, Wedge, PathPatch
)
from shapely.geometry import (
    Point, LineString, Polygon as ShapelyPolygon, MultiPoint
)
from shapely.affinity import translate, scale, rotate
from shapely.ops import unary_union
from shapely.validation import make_valid

from parameter import ShapeParameters
from utils import ShapeUtils

from itertools import combinations
from shapely.geometry import Point, LineString, Polygon as _ShapelyPolygon, MultiPoint, MultiLineString
from shapely.geometry.base import BaseGeometry


def pretty_print_geos_features(res: dict, show_point_limit: int = 20):
    """
    格式化打印 compute_geos_features 的结果。
    - res: compute_geos_features 返回的字典
    - show_point_limit: 列表显示交点/切点/穿过点时的最大项数（其余用 "...(更多 N 项)" 显示）
    """
    if not res:
        print("Empty result.")
        return

    def _show_point_list(name, pts):
        n = len(pts)
        print(f"{name}: {n}")
        if n == 0:
            return
        # 显示前几项
        to_show = pts[:show_point_limit]
        for k, p in enumerate(to_show, start=1):
            print(f"  {k:>2}. {p}")
        if n > show_point_limit:
            print(f"  ... (还有 {n - show_point_limit} 项未显示)")

    # 标题
    print("=" * 72)
    print("Geometry Features Summary".center(72))
    print("=" * 72)
    # 基本计数
    print(f"切点 (tangency_points_count)     : {res.get('tangency_points_count', 0)}")
    print(f"穿过点 / 交叉点 (crossing_points_count): {res.get('crossing_points_count', 0)}")
    print(f"交点总数 (intersection_points_count): {res.get('intersection_points_count', 0)}")
    print(f"部分重叠对数 (partial_overlaps_count) : {res.get('partial_overlaps_count', 0)}")
    print(f"平行边对数 (parallel_edge_pairs_count) : {res.get('parallel_edge_pairs_count', 0)}")
    print("-" * 72)

    # 列出点型交互
    _show_point_list("交点 (intersection_points)", res.get("intersection_points", []))
    print("-" * 72)
    _show_point_list("切点 (tangency_points)", res.get("tangency_points", []))
    print("-" * 72)
    _show_point_list("穿过点 (crossing_points)", res.get("crossing_points", []))
    print("-" * 72)

    # 部分重叠对 (index pairs)
    pairs = res.get("partial_overlaps_pairs", [])
    print(f"部分重叠的 index 对 (partial_overlaps_pairs) : {len(pairs)}")
    if pairs:
        # 输出若干
        limit = 50
        for i, pair in enumerate(pairs[:limit], start=1):
            print(f"  {i:>2}. {pair}")
        if len(pairs) > limit:
            print(f"  ... (还有 {len(pairs)-limit} 项未显示)")
    print("-" * 72)

    # per_geo_info 表格输出
    per_geo = res.get("per_geo_info", [])
    if per_geo:
        # 准备表头和列宽
        headers = ["idx", "n_segments", "straight_chains", "curved_junctions", "n_angles"]
        # 计算列宽（基于内容）
        col_widths = {h: len(h) for h in headers}
        for entry in per_geo:
            for h in headers:
                val = entry.get(h, "")
                s = str(val)
                if len(s) > col_widths[h]:
                    col_widths[h] = len(s)
        # 打印表头
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        sep_line = "-+-".join("-" * col_widths[h] for h in headers)
        print("Per-geometry info:")
        print(header_line)
        print(sep_line)
        # 排序按 idx
        try:
            rows = sorted(per_geo, key=lambda x: int(x.get("idx", 0)))
        except Exception:
            rows = per_geo
        for entry in rows:
            row = " | ".join(str(entry.get(h, "")).ljust(col_widths[h]) for h in headers)
            print(row)
    else:
        print("No per-geometry info available.")
    print("=" * 72)

@staticmethod
def compute_geos_features(geos: List[BaseGeometry],
                          angle_tol_deg: float = 2.0,
                          point_tol: float = 1e-2) -> Dict:
    """
    与原函数一致，但增加了：若 geo A 的某个顶点与 geo B 的某条线段距离 <= point_tol（且投影落在线段上），
    则也判为 tangency（双向检查）。
    """
    # --- 参数与量化 ---
    if point_tol is None or point_tol <= 0:
        point_tol = 1e-6
    angle_tol = math.radians(angle_tol_deg if angle_tol_deg is not None else 2.0)
    if angle_tol <= 0:
        angle_tol = 1e-8
    decimals = max(0, -int(math.floor(math.lg10(point_tol))))

    def quantize_point_obj(p):
        try:
            x = float(p.x)
            y = float(p.y)
        except Exception:
            x, y = float(p[0]), float(p[1])
        return (round(x, decimals), round(y, decimals))

    def is_point_like(g: Optional[BaseGeometry]) -> bool:
        if g is None:
            return False
        if getattr(g, "is_empty", False):
            return False
        gt = getattr(g, "geom_type", "").lower()
        if gt == "point":
            return True
        if gt == "multipoint":
            return True
        if gt in ("linestring", "linertring", "linearring"):
            return getattr(g, "length", 0) <= point_tol
        if gt == "multilinestring":
            return all((getattr(p, "length", 0) <= point_tol) for p in getattr(g, "geoms", []))
        if gt == "polygon":
            return getattr(g, "area", 0) <= point_tol
        if gt == "multipolygon":
            return all((getattr(p, "area", 0) <= point_tol) for p in getattr(g, "geoms", []))
        if gt == "geometrycollection" or gt.startswith("multi"):
            parts = getattr(g, "geoms", [])
            if not parts:
                return False
            return all(is_point_like(part) for part in parts)
        return False

    def extract_point_candidates(g: Optional[BaseGeometry]):
        pts = []
        if g is None or getattr(g, "is_empty", False):
            return pts
        gt = getattr(g, "geom_type", "").lower()
        if gt == "point":
            pts.append(g)
        elif gt == "multipoint":
            for p in g.geoms:
                pts.append(p)
        elif gt in ("linestring", "linearring"):
            if getattr(g, "length", 0) <= point_tol:
                try:
                    pts.append(g.centroid)
                except Exception:
                    try:
                        pts.append(g.representative_point())
                    except Exception:
                        pass
        elif gt == "multilinestring":
            for p in g.geoms:
                if getattr(p, "length", 0) <= point_tol:
                    try:
                        pts.append(p.centroid)
                    except Exception:
                        pass
        elif gt == "polygon":
            if getattr(g, "area", 0) <= point_tol:
                try:
                    pts.append(g.centroid)
                except Exception:
                    pass
        elif gt in ("multipolygon", "geometrycollection") or gt.startswith("multi"):
            for part in getattr(g, "geoms", []):
                pts.extend(extract_point_candidates(part))
        else:
            try:
                pts.append(g.representative_point())
            except Exception:
                pass
        return pts

    def extract_segments(g: Optional[BaseGeometry]):
        segs = []
        if g is None:
            return segs
        gt = getattr(g, "geom_type", "").lower()
        try:
            if gt == "polygon":
                coords = list(g.exterior.coords)
                for k in range(len(coords) - 1):
                    segs.append((coords[k], coords[k + 1]))
                for interior in getattr(g, "interiors", []):
                    icoords = list(interior.coords)
                    for k in range(len(icoords) - 1):
                        segs.append((icoords[k], icoords[k + 1]))
            elif gt == "linestring":
                coords = list(g.coords)
                for k in range(len(coords) - 1):
                    segs.append((coords[k], coords[k + 1]))
            elif gt.startswith("multi") or gt == "geometrycollection":
                for part in getattr(g, "geoms", []):
                    segs.extend(extract_segments(part))
            else:
                try:
                    coords = list(g.coords)
                    for k in range(len(coords) - 1):
                        segs.append((coords[k], coords[k + 1]))
                except Exception:
                    pass
        except Exception:
            pass
        return segs

    def segment_angle(seg: Tuple[Tuple[float, float], Tuple[float, float]]):
        (x0, y0), (x1, y1) = seg
        dx, dy = (x1 - x0), (y1 - y0)
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return None
        ang = math.atan2(dy, dx)
        if ang < 0:
            ang += math.pi
        if ang >= math.pi:
            ang -= math.pi
        return ang

    # --- 新增：点到线段的最短距离及投影参数 ---
    def point_to_segment_distance_and_param(px: float, py: float, seg: Tuple[Tuple[float, float], Tuple[float, float]]):
        (x0, y0), (x1, y1) = seg
        vx = x1 - x0
        vy = y1 - y0
        wx = px - x0
        wy = py - y0
        vlen2 = vx * vx + vy * vy
        if vlen2 == 0:
            # 段退化为点
            dx = px - x0
            dy = py - y0
            return math.hypot(dx, dy), 0.0
        # 投影参数 t
        t = (wx * vx + wy * vy) / vlen2
        # 最近点坐标（可以超出段外）
        projx = x0 + t * vx
        projy = y0 + t * vy
        dx = px - projx
        dy = py - projy
        dist = math.hypot(dx, dy)
        return dist, t

    # --- 初始化统计集合 ---
    tangency_pts = set()
    crossing_pts = set()
    all_inter_pts = set()
    partial_overlaps_pairs = set()

    if not geos:
        return {}

    # 预处理
    valid_geos = []
    for g in geos:
        if g is None:
            continue
        try:
            g_valid = make_valid(g)
        except Exception:
            g_valid = g
        valid_geos.append(g_valid)

    n = len(valid_geos)

    for i, j in combinations(range(n), 2):
        g1 = valid_geos[i]
        g2 = valid_geos[j]
        try:
            if not (g1.intersects(g2) or g1.touches(g2)):
                # 即便没有 intersects/touches，也仍然可能有 "点-线段距离很近" 的情况
                # 所以我们不在这里直接 continue（改为在 intersection 失败后仍做近距离检测）
                pass
        except Exception:
            pass

        inter = None
        try:
            inter = g1.intersection(g2)
        except Exception:
            try:
                inter = make_valid(g1).intersection(make_valid(g2))
            except Exception:
                inter = None

        if inter is None or getattr(inter, "is_empty", False):
            # 交集为空或失败 —— 仍需做点-线段近距离检查（新增规则）
            inter_is_point_like = False
        else:
            try:
                inter_is_point_like = is_point_like(inter)
            except Exception:
                inter_is_point_like = False

        # --- 先保留原有 intersection/touches 逻辑（若有交点则处理） ---
        try:
            if inter is not None and not getattr(inter, "is_empty", False):
                if g1.touches(g2):
                    if inter_is_point_like:
                        pts = extract_point_candidates(inter)
                        if not pts:
                            try:
                                pts = [inter.representative_point()]
                            except Exception:
                                pts = []
                        for p in pts:
                            q = quantize_point_obj(p)
                            tangency_pts.add(q)
                            all_inter_pts.add(q)
                    else:
                        partial_overlaps_pairs.add((i, j))
                else:
                    if g1.intersects(g2) and inter_is_point_like:
                        pts = extract_point_candidates(inter)
                        if not pts:
                            try:
                                pts = [inter.representative_point()]
                            except Exception:
                                pts = []
                        for p in pts:
                            q = quantize_point_obj(p)
                            crossing_pts.add(q)
                            all_inter_pts.add(q)
                    else:
                        itype = getattr(inter, "geom_type", "").lower()
                        if itype in ("linestring", "multilinestring"):
                            if getattr(inter, "length", 0) > point_tol:
                                partial_overlaps_pairs.add((i, j))
                            else:
                                for p in extract_point_candidates(inter):
                                    q = quantize_point_obj(p)
                                    all_inter_pts.add(q)
                        elif itype in ("polygon", "multipolygon"):
                            if getattr(inter, "area", 0) > point_tol:
                                partial_overlaps_pairs.add((i, j))
                            else:
                                for p in extract_point_candidates(inter):
                                    q = quantize_point_obj(p)
                                    all_inter_pts.add(q)
                        else:
                            for part in getattr(inter, "geoms", []):
                                if is_point_like(part):
                                    for p in extract_point_candidates(part):
                                        q = quantize_point_obj(p)
                                        all_inter_pts.add(q)
                                else:
                                    if getattr(part, "length", 0) > point_tol or getattr(part, "area", 0) > point_tol:
                                        partial_overlaps_pairs.add((i, j))
        except Exception:
            try:
                if g1.intersects(g2):
                    inter = g1.intersection(g2)
                    if inter is not None:
                        if getattr(inter, "area", 0) > point_tol or getattr(inter, "length", 0) > point_tol:
                            partial_overlaps_pairs.add((i, j))
                        else:
                            for p in extract_point_candidates(inter):
                                q = quantize_point_obj(p)
                                all_inter_pts.add(q)
            except Exception:
                pass

        # --- 新增补偿规则：点（顶点）与另一几何的任一线段若足够近且投影落在段上 -> tangency ---
        # 抽取 g1 的顶点集合（多边形外环/内环、LineString coords、Point）
        def extract_vertices(g):
            verts = []
            if g is None or getattr(g, "is_empty", False):
                return verts
            gt = getattr(g, "geom_type", "").lower()
            try:
                if gt == "point":
                    verts.append((g.x, g.y))
                elif gt == "multipoint":
                    for p in g.geoms:
                        verts.append((p.x, p.y))
                elif gt == "linestring":
                    for c in list(g.coords):
                        verts.append((float(c[0]), float(c[1])))
                elif gt == "polygon":
                    for c in list(g.exterior.coords):
                        verts.append((float(c[0]), float(c[1])))
                    for interior in getattr(g, "interiors", []):
                        for c in list(interior.coords):
                            verts.append((float(c[0]), float(c[1])))
                elif gt.startswith("multi") or gt == "geometrycollection":
                    for part in getattr(g, "geoms", []):
                        verts.extend(extract_vertices(part))
                else:
                    # 兜底：尝试 coords
                    try:
                        for c in list(g.coords):
                            verts.append((float(c[0]), float(c[1])))
                    except Exception:
                        pass
            except Exception:
                pass
            return verts

        # 抽取 g2 的线段
        segs_g2 = extract_segments(g2)
        # g1 顶点 -> g2 线段
        verts1 = extract_vertices(g1)
        for (px, py) in verts1:
            for seg in segs_g2:
                dist, t = point_to_segment_distance_and_param(px, py, seg)
                # 如果最近点在段上（t in [0,1]）且距离 <= point_tol, 记为 tangency
                if dist <= point_tol and t >= 0.0 and t <= 1.0:
                    # 投影点作为切点位置
                    (x0, y0), (x1, y1) = seg
                    projx = x0 + t * (x1 - x0)
                    projy = y0 + t * (y1 - y0)
                    q = (round(projx, decimals), round(projy, decimals))
                    tangency_pts.add(q)
                    all_inter_pts.add(q)

        # 也做反向检查：g2 顶点 -> g1 线段
        segs_g1 = extract_segments(g1)
        verts2 = extract_vertices(g2)
        for (px, py) in verts2:
            for seg in segs_g1:
                dist, t = point_to_segment_distance_and_param(px, py, seg)
                if dist <= point_tol and t >= 0.0 and t <= 1.0:
                    (x0, y0), (x1, y1) = seg
                    projx = x0 + t * (x1 - x0)
                    projy = y0 + t * (y1 - y0)
                    q = (round(projx, decimals), round(projy, decimals))
                    tangency_pts.add(q)
                    all_inter_pts.add(q)

    # --- per-geo / parallel / chains 与原逻辑保持不变（复制你原来的实现） ---
    per_geo_info = []
    all_segment_angles = []

    for idx, g in enumerate(valid_geos):
        segs = extract_segments(g)
        segs_filtered = []
        for s in segs:
            (x0, y0), (x1, y1) = s
            if abs(x0 - x1) < 1e-12 and abs(y0 - y1) < 1e-12:
                continue
            segs_filtered.append(s)
        segs = segs_filtered

        angles = []
        for s in segs:
            a = segment_angle(s)
            if a is not None:
                angles.append(a)

        straight_chains = 0
        curved_junctions = 0
        if not angles:
            per_geo_info.append({
                "idx": idx,
                "n_segments": len(segs),
                "straight_chains": 0,
                "curved_junctions": 0,
                "n_angles": 0
            })
        else:
            current_chain_len = 1
            for k in range(1, len(angles)):
                da = abs(angles[k] - angles[k - 1])
                da = min(da, math.pi - da)
                if da <= angle_tol:
                    current_chain_len += 1
                else:
                    straight_chains += 1
                    curved_junctions += 1
                    current_chain_len = 1
            if current_chain_len > 0:
                straight_chains += 1
            per_geo_info.append({
                "idx": idx,
                "n_segments": len(segs),
                "straight_chains": straight_chains,
                "curved_junctions": curved_junctions,
                "n_angles": len(angles)
            })

        for a in angles:
            all_segment_angles.append((a, idx))

    buckets = {}
    for ang, geo_idx in all_segment_angles:
        key = int(round(ang / angle_tol))
        buckets.setdefault(key, []).append((ang, geo_idx))
    parallel_pairs = 0
    for key, members in buckets.items():
        m = len(members)
        if m >= 2:
            parallel_pairs += m * (m - 1) // 2

    result = {
        "tangency_points_count": len(tangency_pts),
        "crossing_points_count": len(crossing_pts),
        "intersection_points_count": len(all_inter_pts),
        "tangency_points": list(tangency_pts),
        "crossing_points": list(crossing_pts),
        "intersection_points": list(all_inter_pts),
        "partial_overlaps_count": len(partial_overlaps_pairs),
        "partial_overlaps_pairs": list(partial_overlaps_pairs),
        "parallel_edge_pairs_count": parallel_pairs,
        "per_geo_info": per_geo_info
    }
    return result

class MultiShapeCombinator:
    # ---------------------------
    # 核心辅助：Patch与Shapely转换
    # ---------------------------
    @staticmethod
    def _patch_to_shapely(patch: Patch, resolution: int = 36) -> Optional[ShapelyPolygon]:
        """将Matplotlib Patch转换为Shapely Polygon（近似）"""
        if isinstance(patch, Circle):
            # 圆：Point + buffer
            cx, cy = patch.center
            return Point(cx, cy).buffer(patch.radius, resolution=resolution)
        
        elif isinstance(patch, Rectangle):
            # 矩形：基于左下角坐标和宽高生成Polygon
            x, y = patch.get_x(), patch.get_y()
            w, h = patch.get_width(), patch.get_height()
            coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            return ShapelyPolygon(coords)
        
        elif isinstance(patch, Ellipse):
            # 椭圆：先生成单位圆→缩放→旋转→平移（Shapely无原生椭圆，用多边形近似）
            cx, cy = patch.center
            w, h = patch.width / 2, patch.height / 2  # 半长轴/半短轴
            # 1. 单位圆→缩放至椭圆尺寸→旋转→平移到中心
            ellipse = Point(0, 0).buffer(1, resolution=resolution)
            ellipse = scale(ellipse, xfact=w, yfact=h, origin=(0, 0))
            ellipse = rotate(ellipse, patch.angle, origin=(0, 0))
            ellipse = translate(ellipse, xoff=cx, yoff=cy)
            return ellipse
        
        elif isinstance(patch, Polygon):
            # Matplotlib Polygon→Shapely Polygon
            coords = patch.get_xy()
            if len(coords) < 3:
                return None  # 无效多边形
            return ShapelyPolygon(coords)
        
        elif isinstance(patch, RegularPolygon):
            # 正多边形：计算顶点坐标→Shapely Polygon
            cx, cy = patch.xy if hasattr(patch, 'xy') else patch.center
            radius = patch.radius
            sides = patch.numvertices if hasattr(patch, 'numvertices') else patch.numsides
            angle = patch.orientation  # 初始旋转角度
            # 计算每个顶点的极坐标→直角坐标
            coords = []
            for i in range(sides):
                theta = 2 * math.pi * i / sides + angle
                x = cx + radius * math.cos(theta)
                y = cy + radius * math.sin(theta)
                coords.append((x, y))
            coords.append(coords[0])  # 闭合
            return ShapelyPolygon(coords)
        
        elif isinstance(patch, Wedge):
            # 扇形：圆 + 扇形多边形→交集
            cx, cy = patch.center
            radius = patch.r
            theta1, theta2 = patch.theta1, patch.theta2
            # 1. 生成完整圆
            circle = Point(cx, cy).buffer(radius, resolution=resolution)
            # 2. 生成扇形的"裁剪多边形"（圆心 + 两个半径端点）
            wedge_coords = [(cx, cy)]
            # 角度转弧度，注意Matplotlib Wedge角度从x轴正方向逆时针计算
            for theta in [theta1, theta2]:
                rad = math.radians(theta)
                x = cx + radius * math.cos(rad)
                y = cy + radius * math.sin(rad)
                wedge_coords.append((x, y))
            wedge_coords.append((cx, cy))  # 闭合
            clip_poly = ShapelyPolygon(wedge_coords)
            # 3. 圆与裁剪多边形的交集即为扇形
            wedge = circle.intersection(clip_poly)
            return wedge if isinstance(wedge, ShapelyPolygon) else None
        
        elif isinstance(patch, FancyBboxPatch):
            # 带圆角的矩形：简化为普通矩形（或用buffer模拟圆角，此处取简化方案）
            x, y = patch.get_x(), patch.get_y()
            w, h = patch.get_width(), patch.get_height()
            coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            return ShapelyPolygon(coords)
        
        else:
            # 兜底：基于边界框生成矩形
            bounds = patch.get_extents().bounds  # (xmin, ymin, xmax, ymax)
            x, y, x2, y2 = bounds
            coords = [(x, y), (x2, y), (x2, y2), (x, y2), (x, y)]
            return ShapelyPolygon(coords)

    @staticmethod
    def _shapely_to_patch(geo: ShapelyPolygon, orig_patch: Patch) -> Patch:
        """将Shapely Polygon转回Matplotlib Patch，保留原视觉属性（颜色、线宽）"""
        # 提取原Patch的视觉属性
        # facecolor = orig_patch.get_facecolor()
        facecolor = 'None'
        edgecolor = orig_patch.get_edgecolor()
        linewidth = orig_patch.get_linewidth()
        alpha = orig_patch.get_alpha() or 1.0

        # 处理不同几何类型的逆向转换（优先匹配原Patch类型）
        if isinstance(orig_patch, Circle):
            # 圆：取Shapely圆心和半径
            centroid = geo.centroid
            radius = np.sqrt(geo.area / math.pi)  # 由面积反推半径（近似）
            return Circle(
                (centroid.x, centroid.y), radius,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
            )
        
        elif isinstance(orig_patch, Rectangle):
            # 矩形：取边界框
            xmin, ymin, xmax, ymax = geo.bounds
            w, h = xmax - xmin, ymax - ymin
            return Rectangle(
                (xmin, ymin), w, h,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
            )
        
        elif isinstance(orig_patch, Ellipse):
            # 椭圆：取边界框和旋转角度（简化：用边界框宽高作为椭圆尺寸，忽略旋转误差）
            xmin, ymin, xmax, ymax = geo.bounds
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            w, h = xmax - xmin, ymax - ymin
            return Ellipse(
                (cx, cy), w, h, angle=orig_patch.angle,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
            )
        
        elif isinstance(orig_patch, (Polygon, RegularPolygon, Wedge, FancyBboxPatch)):
            # 多边形类：直接用顶点坐标
            coords = list(geo.exterior.coords)[:-1]  # 去除最后一个重复的闭合点
            return Polygon(
                coords,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
            )
        
        else:
            # 兜底：用PathPatch（兼容所有多边形）
            from matplotlib.path import Path
            coords = list(geo.exterior.coords)
            codes = [Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY]
            path = Path(coords, codes)
            return PathPatch(
                path,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha
            )

    # ---------------------------
    # 功能1：嵌套排列（nested）
    # ---------------------------
    @staticmethod
    def nested(ax: plt.Axes, shapes: List[Patch], shape_params_list: List[ShapeParameters],
              same_center: bool = True, scale_factor: float = 0.4, min_size: float = 0.25) -> List[Dict]:
        """
        Shapely实现：图形从外到内嵌套排列，支持统一中心缩放，基于Shapely的scale/translate实现变换
        """
        if not shapes or len(shapes) != len(shape_params_list):
            return []

        # 1. 转换所有Patch为Shapely几何对象
        shapely_geos = [MultiShapeCombinator._patch_to_shapely(p) for p in shapes]
        # 过滤无效几何（避免后续报错）
        valid_pairs = [(s, g, p) for s, g, p in zip(shapes, shapely_geos, shape_params_list) if g is not None]
        if not valid_pairs:
            return []
        shapes, shapely_geos, shape_params_list = zip(*valid_pairs)

        # 2. 初始化外层图形（基准）
        outer_geo = shapely_geos[0]
        outer_centroid = outer_geo.centroid  # 外层中心（统一中心的基准）
        results = []

        # 3. 循环处理每个图形（从外到内）
        for i, (orig_patch, geo, params) in enumerate(zip(shapes, shapely_geos, shape_params_list)):
            # 计算当前层缩放系数（外层为1.0，内层按scale_factor^i递减）
            current_scale = 1.0 if i == 0 else (scale_factor ** i)
            
            # 3.1 确定目标中心（统一中心/原中心）
            target_centroid = outer_centroid if same_center else geo.centroid

            # 3.2 缩放几何（基于目标中心缩放，避免偏移）
            # Shapely缩放逻辑：先平移到原点→缩放→平移回目标中心
            scaled_geo = scale(
                geo,
                xfact=current_scale, yfact=current_scale,
                origin=(geo.centroid.x, geo.centroid.y)  # 基于原中心缩放
            )

            # 3.3 检查最小尺寸（用边界框面积判断，小于阈值则强制按min_size缩放）
            geo_bounds = scaled_geo.bounds
            geo_width = geo_bounds[2] - geo_bounds[0]
            geo_height = geo_bounds[3] - geo_bounds[1]
            if max(geo_width, geo_height) < min_size:
                # 强制缩放至最小尺寸（基于宽度/高度中较大值）
                resize_ratio = min_size / max(geo_width, geo_height)
                scaled_geo = scale(
                    scaled_geo,
                    xfact=resize_ratio, yfact=resize_ratio,
                    origin=(scaled_geo.centroid.x, scaled_geo.centroid.y)
                )

            # 3.4 平移到目标中心（若统一中心，需调整位置）
            if same_center and i > 0:
                dx = target_centroid.x - scaled_geo.centroid.x
                dy = target_centroid.y - scaled_geo.centroid.y
                scaled_geo = translate(scaled_geo, xoff=dx, yoff=dy)

            # 3.5 转换回Patch并添加到坐标轴
            new_patch = MultiShapeCombinator._shapely_to_patch(scaled_geo, orig_patch)
            ax.add_patch(new_patch)

            # 3.6 更新ShapeParameters（中心、边界框、尺寸）
            new_centroid = scaled_geo.centroid
            new_bounds = scaled_geo.bounds  # (xmin, ymin, xmax, ymax)
            # 尺寸定义：圆形/扇形用半径，其他用边界框宽高
            if isinstance(orig_patch, (Circle, Wedge)):
                new_size = np.sqrt(scaled_geo.area / math.pi)  # 面积反推半径
            else:
                new_size = (new_bounds[2] - new_bounds[0], new_bounds[3] - new_bounds[1])
            
            params.center = (new_centroid.x, new_centroid.y)
            params.bbox = new_bounds
            params.size = new_size
            params.extra_params["scale_factor"] = current_scale

            # 3.7 记录变换结果
            results.append({
                'type': type(orig_patch).__name__,
                'scale': current_scale,
                'new_center': (new_centroid.x, new_centroid.y)
            })

        # 自动调整坐标轴范围
        ax.autoscale_view()
        return results

    # ---------------------------
    # 功能2：邻接排列（adjacent）
    # ---------------------------
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
        更健壮的 adjacent 实现（替代原实现）。

        设计目标：
        - 兼容多种 Shapely 返回类型（Polygon、MultiPolygon、GeometryCollection）
        - 更宽容的距离容差和初始搜索距离（避免初始距离为0导致无法移动）
        - 自动模式下使用角度采样 + 距离倍增策略进行搜索
        - 提高碰撞检测的语义（区分 overlaps/touches/intersects）
        - 保留原始 Patch 的视觉属性（通过 _shapely_to_patch）

        注意：该函数依赖 class 内部的 _patch_to_shapely 和 _shapely_to_patch
        以及外部导入：math, random, numpy as np, shapely.ops.unary_union, shapely.validation.make_valid
        """
        # --- 输入校验 ---
        if not shapes or len(shapes) != len(shape_params_list):
            raise ValueError("shapes 和 shape_params_list 长度必须一致且非空")

        # 转换 Patch -> Shapely（用 sample_n 作为临时分辨率参数）
        shapely_geos = [MultiShapeCombinator._patch_to_shapely(p, resolution=sample_n) for p in shapes]
        valid_pairs = [(s, g, p) for s, g, p in zip(shapes, shapely_geos, shape_params_list) if g is not None]
        if not valid_pairs:
            return []
        shapes, shapely_geos, shape_params_list = zip(*valid_pairs)
        n = len(shapes)

        if adjacency_type == 'auto':
            adjacency_type = random.choice(['edge', 'point'])

        # 辅助：从可能的 Geometry 中选择一个主要的 Polygon（面积最大的）
        from shapely.geometry import MultiPolygon, GeometryCollection, Polygon as _ShapelyPolygon

        def _pick_largest_polygon(g):
            # 处理 make_valid 后可能出现的各种类型
            if g is None:
                return None
            if isinstance(g, _ShapelyPolygon):
                return g
            if isinstance(g, MultiPolygon):
                return max(g.geoms, key=lambda x: x.area)
            if isinstance(g, GeometryCollection):
                polys = [geom for geom in g.geoms if isinstance(geom, _ShapelyPolygon)]
                if polys:
                    return max(polys, key=lambda x: x.area)
                # 退化：取第一个可测面积的几何并缓冲
                for geom in g.geoms:
                    try:
                        p = geom.buffer(0)
                        if isinstance(p, _ShapelyPolygon):
                            return p
                    except Exception:
                        continue
            # 兜底：尝试 unary_union 后取最大 polygon
            try:
                u = unary_union(g)
                if isinstance(u, _ShapelyPolygon):
                    return u
                if isinstance(u, MultiPolygon):
                    return max(u.geoms, key=lambda x: x.area)
            except Exception:
                pass
            return None

        # 初始化放置结构：第一个图形直接作为基准
        placed_geos = []
        placed_centers = []

        # 处理第一个图形：标准化几何并添加
        first_geo = make_valid(shapely_geos[0])
        first_geo = _pick_largest_polygon(first_geo) or shapely_geos[0]
        placed_geos.append(first_geo)
        placed_centers.append((first_geo.centroid.x, first_geo.centroid.y))
        first_shape = MultiShapeCombinator._shapely_to_patch(first_geo, shapes[0])
        # 将第一个原始 patch 添加到 axes（复制视觉属性的 patch 可在之后再做）
        ax.add_patch(first_shape)
        first_params = shape_params_list[0]
        fb = first_geo.bounds
        first_params.center = placed_centers[0]
        first_params.bbox = fb
        first_params.size = (fb[2] - fb[0], fb[3] - fb[1])

        # 随机数生成器（可复现）
        rng = random.Random(0)
        two_pi = 2 * math.pi

        # 主循环：对每个后续图形寻找不重叠且满足 spacing 的位置
        for i in range(1, n):
            orig_patch = shapes[i]
            curr_geo = shapely_geos[i]
            curr_params = shape_params_list[i]

            # 规范几何
            curr_geo = make_valid(curr_geo)
            chosen = _pick_largest_polygon(curr_geo) or curr_geo
            curr_geo = chosen
            if curr_geo is None:
                # 无法处理的几何，跳过
                continue

            # 当前质心和尺寸
            curr_centroid = curr_geo.centroid
            curr_area = curr_geo.area

            # 选取参考几何：已放置图形中最近的
            ref_idx = np.argmin([
                math.hypot(curr_centroid.x - cx, curr_centroid.y - cy)
                for cx, cy in placed_centers
            ])
            ref_geo = placed_geos[ref_idx]
            ref_centroid = ref_geo.centroid

            placed = False
            target_geo = curr_geo

            # 计算合理的初始搜索距离，避免为0
            ref_w = ref_geo.bounds[2] - ref_geo.bounds[0]
            ref_h = ref_geo.bounds[3] - ref_geo.bounds[1]
            curr_w = curr_geo.bounds[2] - curr_geo.bounds[0]
            curr_h = curr_geo.bounds[3] - curr_geo.bounds[1]
            size_based = max((ref_w + curr_w) / 2.0, (ref_h + curr_h) / 2.0)
            initial_dist = max(ref_geo.distance(curr_geo), size_based * 0.5, 1e-3)

            # --- 点对边（point） ---
            if adjacency_type == "point":
                # 构造参考边（分割外环为线段）
                ref_coords = list(ref_geo.exterior.coords)
                ref_edges = [LineString([ref_coords[j], ref_coords[j + 1]]) for j in range(len(ref_coords) - 1)]
                curr_points = [Point(c) for c in list(curr_geo.exterior.coords)[:-1]]

                for curr_pt in curr_points:
                    if placed:
                        break
                    for ref_edge in ref_edges:
                        # 投影到边上并计算当前距离
                        proj_dist_along = ref_edge.project(curr_pt)
                        proj_point = ref_edge.interpolate(proj_dist_along)
                        curr_to_proj = np.array([proj_point.x - curr_pt.x, proj_point.y - curr_pt.y])
                        dist_now = np.linalg.norm(curr_to_proj)
                        # 需要沿 curr->proj 方向移动多少，使距离等于 spacing
                        # needed = spacing - dist_now
                        needed = dist_now - spacing
                        # 如果需要的移动量很小则跳过
                        if abs(needed) <= gap_tol:
                            # 检查是否直接满足且不重叠
                            temp_geo = translate(curr_geo, xoff=0, yoff=0)
                            collision = any(temp_geo.overlaps(g) for g in placed_geos)
                            if not collision:
                                target_geo = temp_geo
                                placed = True
                                break
                            continue
                        # 计算单位矢量（避免 0 除法）
                        if dist_now < 1e-12:
                            # 退化情况：用边的法向（从边中心指向 curr_pt 或反向）
                            edge_mid = ref_edge.interpolate(ref_edge.length / 2.0)
                            vec = np.array([curr_pt.x - edge_mid.x, curr_pt.y - edge_mid.y])
                            norm = np.linalg.norm(vec) or 1.0
                            unit = vec / norm
                        else:
                            unit = curr_to_proj / dist_now
                        # 平移方向：沿 unit 方向移动 needed
                        temp_geo = translate(curr_geo, xoff=unit[0] * needed, yoff=unit[1] * needed)
                        # 检查与其他已放置的关系：不允许 overlaps
                        if not any(temp_geo.overlaps(g) for g in placed_geos):
                            print('-' * 30)
                            # 并且与参考边的距离接近 spacing
                            if abs(temp_geo.distance(ref_edge) - spacing) <= max(gap_tol, 1e-4):
                                target_geo = temp_geo
                                placed = True
                                break
                    if placed:
                        break

            # --- 边对边（edge） ---
            elif adjacency_type == "edge":
                # 基于边界框简化：匹配水平或垂直边
                rb = ref_geo.bounds
                cb = curr_geo.bounds
                # 构造四条边的线段
                ref_h_edges = [LineString([(rb[0], rb[1]), (rb[2], rb[1])]), LineString([(rb[0], rb[3]), (rb[2], rb[3])])]
                ref_v_edges = [LineString([(rb[0], rb[1]), (rb[0], rb[3])]), LineString([(rb[2], rb[1]), (rb[2], rb[3])])]
                curr_h_edges = [LineString([(cb[0], cb[1]), (cb[2], cb[1])]), LineString([(cb[0], cb[3]), (cb[2], cb[3])])]
                curr_v_edges = [LineString([(cb[0], cb[1]), (cb[0], cb[3])]), LineString([(cb[2], cb[1]), (cb[2], cb[3])])]

                # 尝试匹配水平-水平、垂直-垂直
                for ref_edge in ref_h_edges + ref_v_edges:
                    if placed:
                        break
                    # 判断边类型
                    x0, y0 = ref_edge.coords[0]
                    x1, y1 = ref_edge.coords[-1]
                    is_horizontal = abs(y0 - y1) < 1e-8
                    curr_edges = curr_h_edges if is_horizontal else curr_v_edges

                    for curr_edge in curr_edges:
                        # 计算边到边的最短距离
                        dist_now = ref_edge.distance(curr_edge)
                        needed = spacing - dist_now
                        # 方向：把 curr_edge 沿 ref_edge 的法线方向平移 needed
                        # 取 ref_edge 的单位法向
                        dx_e = x1 - x0
                        dy_e = y1 - y0
                        edge_len = math.hypot(dx_e, dy_e) or 1.0
                        # 法向（旋转90度）
                        nx, ny = -dy_e / edge_len, dx_e / edge_len
                        temp_geo = translate(curr_geo, xoff=nx * needed, yoff=ny * needed)
                        if not any(temp_geo.overlaps(g) for g in placed_geos):
                            # 检查与参考边的距离
                            if abs(temp_geo.distance(ref_edge) - spacing) <= max(gap_tol, 1e-4):
                                target_geo = temp_geo
                                placed = True
                                break
                    if placed:
                        break

            # --- 自动（auto / fallback） ---
            if not placed or adjacency_type == "auto":
                print('-' * 30)
                # 角度列表（转换成 list 后 shuffle）
                angles = list(np.linspace(angle, angle + two_pi, sample_n, endpoint=False))
                if mode == "random":
                    rng.shuffle(angles)

                scale_step = 1.0
                max_scale = 50
                while scale_step <= max_scale and not placed:
                    for theta in angles:
                        dx_dir = math.cos(theta)
                        dy_dir = math.sin(theta)
                        translate_dist = initial_dist * scale_step
                        temp_geo = translate(curr_geo, xoff=dx_dir * translate_dist, yoff=dy_dir * translate_dist)

                        # 计算与参考图形的距离
                        ref_dist = temp_geo.distance(ref_geo)
                        # 对 spacing 的容忍条件
                        if spacing == 0.0:
                            # 允许接触但不允许面积重叠
                            collision = any(temp_geo.overlaps(g) for g in placed_geos)
                            if not collision and ref_dist <= max(gap_tol, 1e-4):
                                target_geo = temp_geo
                                placed = True
                                break
                        else:
                            if abs(ref_dist - spacing) <= max(gap_tol, 1e-3) and not any(temp_geo.overlaps(g) for g in placed_geos):
                                target_geo = temp_geo
                                placed = True
                                break
                    scale_step *= 1.4

            # 兜底：如果仍未放置，沿参考中心方向强制移开直到不重叠
            if not placed:
                dir_vec = (curr_centroid.x - ref_centroid.x, curr_centroid.y - ref_centroid.y)
                dir_norm = math.hypot(dir_vec[0], dir_vec[1]) or 1.0
                dir_unit = (dir_vec[0] / dir_norm, dir_vec[1] / dir_norm)
                translate_dist = initial_dist + abs(spacing)
                attempts = 0
                while attempts < 200:
                    temp_geo = translate(curr_geo, xoff=dir_unit[0] * translate_dist, yoff=dir_unit[1] * translate_dist)
                    if not any(temp_geo.overlaps(g) for g in placed_geos):
                        target_geo = temp_geo
                        placed = True
                        break
                    translate_dist *= 1.25
                    attempts += 1

            # 确保返回合法 Polygon
            target_geo = make_valid(target_geo)
            pick = _pick_largest_polygon(target_geo) or target_geo
            if pick is None:
                pick = curr_geo
            target_geo = pick

            # # --- 在将 target_geo 转回 Patch 并添加之前：计算特征 ---
            # features = compute_pairwise_features(
            #     ref_geo, target_geo, patch_a=shapes[ref_idx], patch_b=orig_patch,
            #     angle_tol_deg=5.0, parallel_angle_tol_deg=5.0, parallel_distance_ratio=0.05
            # )
            # # 将特征写入当前 shape_params（确保 extra_params 存在）
            # if not hasattr(curr_params, "extra_params") or curr_params.extra_params is None:
            #     curr_params.extra_params = {}
            # curr_params.extra_params["adjacency_features"] = features

            # 转换回 Patch 并添加到 axes
            new_patch = MultiShapeCombinator._shapely_to_patch(target_geo, orig_patch)
            ax.add_patch(new_patch)

            # 更新参数
            new_centroid = target_geo.centroid
            new_bounds = target_geo.bounds
            if isinstance(orig_patch, (Circle, Wedge)):
                new_size = np.sqrt(target_geo.area / math.pi)
            else:
                new_size = (new_bounds[2] - new_bounds[0], new_bounds[3] - new_bounds[1])

            curr_params.center = (new_centroid.x, new_centroid.y)
            curr_params.bbox = new_bounds
            curr_params.size = new_size

            placed_geos.append(target_geo)
            placed_centers.append((new_centroid.x, new_centroid.y))
        
        # print(placed_geos)
        pretty_print_geos_features(compute_geos_features(placed_geos))

        ax.autoscale_view()
        return placed_centers


    # ---------------------------
    # 功能3：相交排列（intersecting）
    # ---------------------------
    @staticmethod
    def intersecting(ax: mpl.axes.Axes, 
                    shapes: List[Patch], 
                    shape_params_list: List[ShapeParameters], 
                    overlap_style: str = "random") -> List[Tuple[float, float]]:
        """
        Shapely实现：图形实质性相交排列，基于Shapely的intersects/intersection判断重叠
        """
        if not shapes or len(shapes) != len(shape_params_list):
            raise ValueError("shapes与shape_params_list长度必须一致")

        # 1. 转换Patch为Shapely几何，过滤无效对象
        shapely_geos = [MultiShapeCombinator._patch_to_shapely(p) for p in shapes]
        valid_pairs = [(s, g, p) for s, g, p in zip(shapes, shapely_geos, shape_params_list) if g is not None]
        if not valid_pairs:
            return []
        shapes, shapely_geos, shape_params_list = zip(*valid_pairs)
        n = len(shapes)

        # 2. 初始化基准图形（第一个图形）
        placed_geos = [shapely_geos[0]]
        placed_centers = [(shapely_geos[0].centroid.x, shapely_geos[0].centroid.y)]
        # 添加到坐标轴并更新参数
        ax.add_patch(shapes[0])
        first_params = shape_params_list[0]
        first_bounds = shapely_geos[0].bounds
        first_params.center = placed_centers[0]
        first_params.bbox = first_bounds
        first_params.size = (first_bounds[2]-first_bounds[0], first_bounds[3]-first_bounds[1])

        # 3. 处理后续图形（确保与参考图形相交，且不与其他图形过度重叠）
        rng = random.Random(42)  # 固定种子
        two_pi = 2 * math.pi
        min_overlap_ratio = 0.05  # 最小重叠面积比例（避免仅接触）

        for i in range(1, n):
            orig_patch = shapes[i]
            curr_geo = shapely_geos[i]
            curr_params = shape_params_list[i]
            curr_centroid = curr_geo.centroid
            curr_area = curr_geo.area

            # 3.1 选择参考图形（优先最近的已放置图形）
            ref_idx = np.argmin([
                math.hypot(curr_centroid.x - cx, curr_centroid.y - cy)
                for cx, cy in placed_centers
            ])
            ref_geo = placed_geos[ref_idx]
            ref_centroid = ref_geo.centroid
            ref_area = ref_geo.area

            target_geo = None
            max_attempts = 50  # 最大尝试次数（避免无限循环）
            attempt = 0

            # 3.2 按相交模式计算目标位置
            while attempt < max_attempts and target_geo is None:
                if overlap_style == "random":
                    # 随机模式：随机平移，确保与参考图形相交且重叠面积达标
                    # 平移范围：参考图形边界框内（确保高概率相交）
                    ref_bounds = ref_geo.bounds
                    dx = rng.uniform(ref_bounds[0] - curr_geo.bounds[2], ref_bounds[2] - curr_geo.bounds[0])
                    dy = rng.uniform(ref_bounds[1] - curr_geo.bounds[3], ref_bounds[3] - curr_geo.bounds[1])
                    temp_geo = translate(curr_geo, xoff=dx, yoff=dy)

                elif overlap_style == "center":
                    # 中心模式：向参考中心偏移，确保重叠
                    offset_ratio = rng.uniform(0.3, 0.7)  # 偏移比例（0=完全重叠，1=边缘）
                    theta = rng.uniform(0, two_pi)  # 随机偏移方向
                    # 偏移距离：参考图形尺寸的offset_ratio倍
                    ref_width = ref_bounds[2] - ref_bounds[0]
                    ref_height = ref_bounds[3] - ref_bounds[1]
                    offset_dist = math.hypot(ref_width, ref_height) * offset_ratio * 0.5
                    dx = math.cos(theta) * offset_dist
                    dy = math.sin(theta) * offset_dist
                    # 平移到参考中心附近
                    temp_geo = translate(
                        curr_geo,
                        xoff=ref_centroid.x - curr_centroid.x + dx,
                        yoff=ref_centroid.y - curr_centroid.y + dy
                    )

                else:
                    raise ValueError(f"不支持的相交模式: {overlap_style}")

                # 3.3 验证条件：1. 与参考图形相交；2. 重叠面积达标；3. 不与其他图形过度重叠
                if temp_geo.intersects(ref_geo):
                    # 计算重叠面积（确保不小于最小比例）
                    overlap = temp_geo.intersection(ref_geo)
                    overlap_area = overlap.area if isinstance(overlap, ShapelyPolygon) else 0.0
                    min_overlap_area = min(curr_area, ref_area) * min_overlap_ratio
                    # 检查重叠面积和与其他图形的关系
                    if overlap_area >= min_overlap_area and not any(
                        temp_geo.intersection(g).area > min_overlap_area for g in placed_geos if g != ref_geo
                    ):
                        target_geo = temp_geo

                attempt += 1

            # 3.4 兜底：若未找到位置，强制与参考图形中心重叠（确保相交）
            if target_geo is None:
                dx = ref_centroid.x - curr_centroid.x
                dy = ref_centroid.y - curr_centroid.y
                target_geo = translate(curr_geo, xoff=dx, yoff=dy)

            # 3.5 修复拓扑错误并转换回Patch
            target_geo = make_valid(target_geo)
            new_patch = MultiShapeCombinator._shapely_to_patch(target_geo, orig_patch)
            ax.add_patch(new_patch)

            # 3.6 更新参数和已放置列表
            new_centroid = target_geo.centroid
            new_bounds = target_geo.bounds
            if isinstance(orig_patch, (Circle, Wedge)):
                new_size = np.sqrt(target_geo.area / math.pi)
            else:
                new_size = (new_bounds[2]-new_bounds[0], new_bounds[3]-new_bounds[1])
            
            curr_params.center = (new_centroid.x, new_centroid.y)
            curr_params.bbox = new_bounds
            curr_params.size = new_size
            placed_geos.append(target_geo)
            placed_centers.append((new_centroid.x, new_centroid.y))

        # 自动调整坐标轴
        ax.autoscale_view()
        return placed_centers