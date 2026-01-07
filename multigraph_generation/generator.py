from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, TypeAlias, Union, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import transforms as mtransforms
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
from matplotlib.spines import Spine
from matplotlib.patches import Patch, PathPatch
from shapely.geometry import Polygon, Point, MultiPolygon, GeometryCollection
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, Collection

from config import Config
from shapes import BaseShapes
from logger import setup_logger
from parameter import ShapeParameters
from utils import ShapeUtils
from style import StyleEnhancer
from single_variants import SingleShapeVariants
from multi_combinator import MultiShapeCombinator
from check import check_axes_artists_inside

BoundsType: TypeAlias = Union[Tuple[float, float], Tuple[float, float, float, float]]


@dataclass
class GenerationRecord:
    """存储一次生成过程的所有记录"""
    generation_id: str  # 生成ID
    timestamp: str  # 时间戳
    seed: Optional[int] = None  # 随机种子
    mode: str = "random"  # 生成模式
    shape_count: int = 0  # 图形数量
    bounds: BoundsType = (-5.0, 5.0)  # 边界范围
    global_scale: float = 1.3  # 全局缩放
    shapes: List[ShapeParameters] = field(default_factory=list)  # 图形参数列表

# ---------------------------
# 几何生成器主类
# ---------------------------
class GeometryGenerator:
    """几何图形生成器（整合所有组件，提供统一生成接口）"""
    def __init__(self, 
                 bounds: BoundsType = Config.DEFAULT_BOUNDS,
                 global_scale: float = Config.DEFAULT_GLOBAL_SCALE,
                 log_level: str = "INFO",
                ) -> None:
        self.bounds = bounds
        self.global_scale = float(global_scale)
        self.shape_types = [
            ("circle", BaseShapes.circle),
            ("ellipse", BaseShapes.ellipse),
            ("rectangle", BaseShapes.rectangle),
            ("regular_polygon", BaseShapes.regular_polygon),
            ("sector", BaseShapes.sector),
        ]

        # 初始化日志
        self.logger = setup_logger(
            log_level=log_level
        )
        self.logger.info(
            "生成器初始化：边界=%s, 全局缩放=%.2f, 日志级别=%s",
            bounds, global_scale, log_level
        )
        
        # 记录生成历史
        self.generation_history: List[GenerationRecord] = []

    def  _generate_base_shapes(self, ax: plt.Axes, count: int = 3, attempts: int = Config.DEFAULT_GENERATE_ATTEMPTS, mode: str = 'random') -> Tuple[List[Patch], List[ShapeParameters]]:
        """生成基础形状（确保在边界内）并记录参数"""
        shapes: List[Patch] = []
        shape_params: List[ShapeParameters] = []
        tries = 0
        self.logger.debug("生成基础形状：目标数量=%d, 最大尝试=%d", count, attempts)

        while len(shapes) < count and tries < attempts:
            tries += 1
            name, func = random.choice(self.shape_types)
            
            # 生成中心（在边界内）
            if len(self.bounds) == 2:
                center = ShapeUtils.random_point(self.bounds)
            else:
                x0, x1, y0, y1 = self.bounds
                center = ShapeUtils.random_point((x0, x1, y0, y1))

            if mode == 'random':
                raw_size = 3.0
            else:
                raw_size = ShapeUtils.random_size(2.2, 3.2)
            # 生成尺寸（应用全局缩放）
            size = raw_size * self.global_scale

            # 创建形状
            try:
                if name == "circle":
                    shape = func(center, radius=size / 2)
                elif name == "ellipse":
                    shape = func(center, width=size, height=size * 0.6, angle=ShapeUtils.random_rotation())
                elif name == "rectangle":
                    round_corner = random.uniform(0.0, 0.5) if random.random() < 0.3 else 0.0
                    shape = func(
                        (center[0] - size / 2, center[1] - size * 0.6 / 2),
                        width=size, height=size * 0.6, round_corner=round_corner
                    )
                elif name == "regular_polygon":
                    shape = func(center, num_edges=random.randint(3, 8), radius=size / 2)
                else:  # sector
                    shape = func((0, 0), radius=size / 2, 
                                theta1=random.uniform(0, 180), theta2=random.uniform(90, 360))
            except Exception as e:
                self.logger.exception("创建%s失败：%s", name, e)
                continue

            # # 检查边界
            # if not ShapeUtils.check_bounds(shape, self.bounds):
            #     self.logger.debug("跳过%s：中心=%s, 尺寸=%.2f（超出边界）", name, center, size)
            #     continue

            # 记录形状参数
            shape_id = f"{name}_{len(shapes)}"
            params = ShapeUtils.get_shape_parameters(shape, ax, shape_id)
            shapes.append(shape)
            shape_params.append(params)
            self.logger.debug("创建%s：边界=%s", name, ShapeUtils.get_bbox(shape, ax))

        if not shapes:
            self.logger.warning("生成基础形状失败：%d次尝试后无有效形状", tries)
        else:
            self.logger.info("生成基础形状：成功创建%d个（尝试%d次）", len(shapes), tries)

        return shapes, shape_params

    def _center_shapes_to_canvas(self, shapes: List[Patch], shape_params: List[ShapeParameters]) -> None:
        """将形状组居中到画布中心并更新参数记录"""
        try:

            # 目标中心（边界中心）
            if len(self.bounds) == 2:
                target_center = ((self.bounds[0] + self.bounds[1]) / 2.0, (self.bounds[0] + self.bounds[1]) / 2.0)
            else:
                target_center = ((self.bounds[0] + self.bounds[1]) / 2.0, (self.bounds[2] + self.bounds[3]) / 2.0)

            # 平移
            for shape in shapes:
                try:
                    if isinstance(shape, Circle):
                        shape.center = (target_center[0], target_center[1])
                    
                    elif isinstance(shape, Ellipse):
                        shape.center = (target_center[0], target_center[1])
                    
                    elif isinstance(shape, (Rectangle, FancyBboxPatch)):
                        shape.set_x(target_center[0])
                        shape.set_y(target_center[1])
                    
                    elif isinstance(shape, Polygon):
                        verts = shape.get_xy()
                        shape.set_xy(np.array([target_center[0], target_center[1]], dtype=np.float32))
                    
                    elif isinstance(shape, RegularPolygon):
                        r = float(getattr(shape, "radius", getattr(shape, "r", 0.0)))
                        ShapeUtils._reposition_regular_polygon(shape, (target_center[0], target_center[1]), r)
                    
                    elif isinstance(shape, Wedge):
                        shape.center = [target_center[0], target_center[1]]
                    
                    else:
                        # 通用平移：修改顶点或变换矩阵
                        try:
                            verts = shape.get_xy()
                            shape.set_xy(verts + np.array([target_center[0], target_center[1]], dtype=np.float32))
                        except Exception:
                            self.logger.exception("形状居中失败：%s", e)
                except Exception as e:
                    self.logger.exception("形状居中失败：%s", e)
        except Exception as e:
            self.logger.exception("形状居中失败：%s", e)

    def _render_single_shape(self, ax: mpl.axes.Axes, shape: Patch, params: ShapeParameters) -> None:
        """渲染单个形状（添加样式、装饰等）并更新参数记录"""
        StyleEnhancer.get_random_style(shape, params)
        
        # d对于两种变式采用不同的时间点add_patch，单形状装饰在最后add_patch，单形状掩码在开头add_patch
        # 随机装饰
        if random.random() < 0.7:
            self.logger.debug("添加单形状装饰")
            decoration_style = random.choice(["radial", "grid", "random", "polygon"])
            SingleShapeVariants.add_internal_decoration(ax, shape, params, style=decoration_style)
            if shape not in ax.patches:   
                ax.add_patch(shape)
        
        # 随机掩码
        else:
            ax.add_patch(shape)
            self.logger.debug("添加单形状掩码")
            mask_type = random.choice(["cut", "replace_boundary"])
            SingleShapeVariants.apply_mask(ax, shape, params, mask_type=mask_type)
        
        # # 随机变形（仅多边形）
        # if random.random() < 0.3 and isinstance(shape, Polygon):
        #     self.logger.debug("变形单形状边缘")
        #     SingleShapeVariants.deform_edge(shape)
        #     # 更新变形后的参数
        #     params.center = ShapeUtils.get_center(shape)
        #     params.bbox = ShapeUtils.get_bbox(shape)
        
        
        
        self.logger.info("渲染单形状：类型=%s, 边界=%s", type(shape), ShapeUtils.get_bbox(shape, ax))

    def _render_multi_shapes(self, ax: mpl.axes.Axes, shapes: List[Patch], shape_params: List[ShapeParameters], mode: str) -> None:
        """渲染多形状组合（嵌套/相邻/交叉/网格）并更新参数记录"""
        combo_mode = random.choice(["nested", "adjacent", "intersecting"]) if mode == "random" else mode
        self.logger.info("渲染多形状组合：模式=%s", combo_mode)

        # 基础样式
        line_width = random.uniform(1.5, 2.0)
        for s, params in zip(shapes, shape_params):
            StyleEnhancer.get_random_style(s, params, line_width=line_width)
            if random.random() < 0.4:
                StyleEnhancer.rotate(s, ax, params)

        # 组合排列
        if combo_mode == "nested":
            MultiShapeCombinator.nested(ax, shapes, shape_params, same_center=True)
        elif combo_mode == "adjacent":
            MultiShapeCombinator.adjacent(ax, shapes, shape_params, mode="random")
        elif combo_mode == "intersecting":
            MultiShapeCombinator.intersecting(ax, shapes, shape_params, overlap_style="random")

    def _save_figure(self, fig: plt.Figure, save_path: str, dpi: int) -> None:
        """保存图像（自动创建目录）"""
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, pad_inches=0.06)
            self.logger.info("保存图像：%s", save_path)
        except Exception as e:
            self.logger.exception("保存图像失败：%s", e)

    def center_combined_shapes(self, ax: plt.Axes, shapes: list, include_ax_children: bool = True, extra_types: tuple = None):
        """
        将 shapes 的组合（按 data 坐标）平移到 ax 的 data 坐标中心。
        同时可选择把 ax 上的其它可移动 artists 一并纳入（例如 Line2D, PathCollection 等）。

        Args:
            ax: 目标 Axes
            shapes: 你显式构建的 shapes 列表（可以包含 Patch, Line2D, Collection 等）
            include_ax_children: 是否将 ax.get_children() 中新增且属于 extra_types 的 artist 一并考虑并移动
            extra_types: 可选的额外类型元组，用于匹配 ax children；默认包含 Line2D, Patch, Collection
        Returns:
            moved: 最终被移动的 artist 列表（shapes + 新增匹配到的 children）
        """
        if not shapes and not include_ax_children:
            return []

        # 默认匹配类型（可覆盖）
        if extra_types is None:
            extra_types = (Line2D, Patch, PathCollection, Collection)

        # 获取 ax children（拷贝列表，避免迭代期间修改）
        ax_children = list(ax.get_children())

        # 先把 shapes 中已存在的对象标准化为列表引用（避免传入参数为参数描述等）
        base_shapes = list(shapes) if shapes is not None else []

        # 如果 include_ax_children，找出那些在 ax.children 中但不在 base_shapes 中的、匹配 extra_types 的 artist
        extras = []
        if include_ax_children:
            for child in ax_children:
                # # 排除坐标轴本身的装饰（比如 xaxis, yaxis, spines, texts 等）
                # if child in base_shapes:
                #     continue
                # 只收集用户关心的可移动类型
                if isinstance(child, extra_types) and not isinstance(child, Spine):
                    # 有时会把 Axes 的 background patch 等也算进来；过滤掉不是数据可见/可移动的（例如 Axis artist）
                    # 简单策略：只接受有 get_window_extent / get_extent 或 get_xdata/get_ydata 的 artist
                    extras.append(child)

        # 合并最终参与居中的 artist 列表（注意保持原顺序：先 base_shapes，再 extras）
        all_artists = extras

        if not all_artists:
            return []

        # 1. 计算每个 artist 在 data 坐标下的 bbox（依赖 ShapeUtils.get_bbox 支持 Line2D / Collection）
        bboxes = []
        for art in all_artists:
            # try:
            print(art)
            bb = ShapeUtils.get_bbox(art, ax)  # 应返回 data-space Bbox
            bboxes.append(bb)
            # print(bb)
            # except Exception as e:
            #     # 如果某个 artist 无法计算 bbox，记录并跳过（但建议打印警告）
            #     logging.debug("compute bbox failed for %s: %s", type(art), e)

        # 如果没有可用 bbox，则直接返回
        if not bboxes:
            return []

        # 合并 bbox
        xmin = min(b.x0 for b in bboxes)
        ymin = min(b.y0 for b in bboxes)
        xmax = max(b.x1 for b in bboxes)
        ymax = max(b.y1 for b in bboxes)

        # 2. 组合中心（data 坐标）
        combined_center_x = (xmin + xmax) / 2.0
        combined_center_y = (ymin + ymax) / 2.0
        print(combined_center_x)
        print(combined_center_y)

        # 3. 画布中心（data 坐标）
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        canvas_center_x = (ax_xlim[0] + ax_xlim[1]) / 2.0
        canvas_center_y = (ax_ylim[0] + ax_ylim[1]) / 2.0
        # print(canvas_center_x)
        # print(canvas_center_y)

        # 4. 需要平移的量（data 单位）
        dx = canvas_center_x - combined_center_x
        dy = canvas_center_y - combined_center_y

        # 5. 对每个 artist 应用平移（在 data 单位）
        moved = []
        for art in all_artists:
            try:
                ShapeUtils.translate_shape(art, dx, dy, ax)
                moved.append(art)
            except Exception as e:
                logging.debug("translate failed for %s: %s", type(art), e)
                # 如果直接失败，尝试 fallback：把 data dx/dy 转 display 并叠加 transform
                try:
                    p0 = ax.transData.transform((0.0, 0.0))
                    p1 = ax.transData.transform((dx, dy))
                    tx, ty = p1[0] - p0[0], p1[1] - p0[1]
                    t = mtransforms.Affine2D().translate(tx, ty)
                    try:
                        art.set_transform(t + art.get_transform())
                        moved.append(art)
                    except Exception:
                        art.set_transform(t + ax.transData)
                        moved.append(art)
                except Exception as e2:
                    logging.debug("final fallback failed for %s: %s", type(art), e2)

        # 6. 刷新（延迟重绘）
        try:
            ax.figure.canvas.draw_idle()
        except Exception:
            try:
                ax.figure.canvas.draw()
            except Exception:
                pass

        return moved

    # optional helper used above if you had such function originally
    @staticmethod
    def _reposition_regular_polygon(shape: RegularPolygon, center: tuple, radius: float):
        """
        如果你有对 RegularPolygon 的重定位函数，可以放在这里；
        这是一个示例实现（假设 shape._regular_verts 或类似字段可直接修改）。
        否则，上面的代码会尝试通过 set transform / set_xy 回退处理。
        """
        # 这是示范性占位；具体实现依赖于 matplotlib RegularPolygon 的内部实现
        try:
            # recompute vertices based on n, radius, orientation, center
            n = shape.numvertices
            orientation = getattr(shape, "orientation", 0.0)
            cx, cy = center
            angles = orientation + np.linspace(0, 2 * np.pi, n, endpoint=False)
            verts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
            # 某些 RegularPolygon 没有 set_xy；则直接尝试赋值
            if hasattr(shape, "set_xy"):
                shape.set_xy(verts)
            else:
                # 尝试写入私有字段（视 matplotlib 版本而定）
                if hasattr(shape, "xy"):
                    shape.xy = verts
        except Exception:
            pass


    def _save_parameters(self, record: GenerationRecord, save_path: str) -> None:
        """保存参数记录到JSON文件"""
        try:
            # 创建保存目录
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
            # 转换为字典以便JSON序列化
            def dataclass_to_dict(obj):
                if isinstance(obj, (list, tuple)):
                    return [dataclass_to_dict(item) for item in obj]
                if not hasattr(obj, '__dict__'):
                    return obj
                result = obj.__dict__.copy()
                # 移除无法序列化的属性
                if 'patch' in result:
                    del result['patch']
                return result
            
            # 自定义编码器：读取 __dict__ 获取所有属性
            def shape_encoder(obj):
                if isinstance(obj, ShapeParameters):
                    return obj.__dict__  # 直接返回所有属性的键值对
                if isinstance(obj, Bbox):
                    return obj.__dict__  # 直接返回所有属性的键值对
                if isinstance(obj, Line2D):
                    return obj.__dict__  # 直接返回所有属性的键值对
                if isinstance(obj, np.ndarray):
                    return list(obj)  # 直接返回所有属性的键值对
                # raise TypeError(f"无法序列化类型：{type(obj)}")
                return ''
            
            record_dict = dataclass_to_dict(record)
            
            # 保存到文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(record_dict, f, ensure_ascii=False, indent=2, default=shape_encoder)
            self.logger.info("保存参数记录：%s", save_path)
        except Exception as e:
            self.logger.exception("保存参数记录失败：%s", e)

    def generate(self, 
                mode: str = "random",
                save_path: Optional[str] = None,
                params_save_path: Optional[str] = None,
                dpi: int = Config.DEFAULT_DPI,
                seed: Optional[int] = None,
                center_on_canvas: bool = True) -> GenerationRecord:
        """生成几何图形并可选保存
        
        Args:
            mode: 生成模式（random/nested/adjacent/intersecting）
            save_path: 图像保存路径（None则不保存）
            params_save_path: 参数记录保存路径（None则不保存）
            dpi: 图像DPI
            seed: 随机种子（确保可复现）
            center_on_canvas: 是否居中到画布
        
        Returns:
            生成记录对象，包含所有图形的参数信息
        """
        import uuid
        from datetime import datetime
        
        # 生成唯一ID和时间戳
        generation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 设置随机种子
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed) & 0xFFFFFFFF)  # numpy种子取低32位
            self.logger.info("生成图像：使用种子=%s", seed)

        self.logger.info(
            "生成图像：ID=%s, 模式=%s, 保存路径=%s, DPI=%d, 居中=%s",
            generation_id, mode, save_path, dpi, center_on_canvas
        )

        # 创建画布
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("#f8f9fa")
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])

        # 生成基础形状和参数记录
        shape_count = 1 if mode == "random" else random.randint(2, 3)
        self.logger.debug("生成图像：选择形状数量=%d", shape_count)
        shapes, shape_params = self._generate_base_shapes(ax, shape_count, mode=mode)
        if not shapes:
            self.logger.error("生成图像失败：无有效形状")
            plt.close(fig)
            # 返回空记录
            return GenerationRecord(
                generation_id=generation_id,
                timestamp=timestamp,
                seed=seed,
                mode=mode
            )

        # 居中形状
        if center_on_canvas:
            self._center_shapes_to_canvas(shapes, shape_params)
    
        for shape in shapes:
            print(shape)
        # 渲染形状
        if len(shapes) == 1:
            self._render_single_shape(ax, shapes[0], shape_params[0])
        else:
            self._render_multi_shapes(ax, shapes, shape_params, mode)


        # for shape in ax.get_children():
        #     print(shape)

        # 将组合图形移到画布中心
        self.center_combined_shapes(ax, shapes)



        # 创建生成记录
        record = GenerationRecord(
            generation_id=generation_id,
            timestamp=timestamp,
            seed=seed,
            mode=mode,
            shape_count=len(shapes),
            bounds=self.bounds,
            global_scale=self.global_scale,
            shapes=shape_params
        )
        
        # 保存到历史记录
        self.generation_history.append(record)

        # 保存图像
        if save_path:
            self._save_figure(fig, save_path, dpi)

        # 保存参数记录
        if params_save_path:
            self._save_parameters(record, params_save_path)

        res = check_axes_artists_inside(ax)
        if not res["all_inside"]:
            print("有图形不在画布内：")
            for item in res["out_of_bounds"]:
                art = item["artist"]
                reason = item["reason"]
                bbox = item["bbox_data"]
                print(type(art), reason, bbox)
        else:
            print("所有检查的图形都在画布内")
        
        plt.close(fig)
        self.logger.debug("生成图像完成：关闭画布，ID=%s", generation_id)
        
        return record