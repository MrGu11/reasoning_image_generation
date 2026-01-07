# shapes_with_external.py
"""
cv2 (OpenCV) implementation of Shape with optional anti-aliasing modes.

Extended: 支持更多形状：pentagon, hexagon, plus, heart, crescent, rounded_square
Added: 支持绘制外部图像/纹理 overlay（文件路径 / PIL.Image / np.ndarray）
修改：统一旋转方式为「顺时针为正角度」
"""
import math
from typing import Tuple, List, Union, Optional
import cv2
import numpy as np
import os

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# optional svg rasterizer (not required)
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

from utils import rand_color

ImageLike = Union[np.ndarray, 'PIL.Image.Image']  # type: ignore

# ------------------ helpers ------------------
def to_cv2(img: ImageLike) -> np.ndarray:
    if PIL_AVAILABLE and isinstance(img, Image.Image):
        arr = np.array(img)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()
    else:
        raise ValueError("Unsupported image type for to_cv2")


def from_cv2(img: np.ndarray) -> 'PIL.Image.Image':
    if not PIL_AVAILABLE:
        raise RuntimeError('PIL not available')
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _to_bgr_tuple(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

# ------------------ soft-mask helper ------------------
def draw_soft_filled_poly(img: np.ndarray, pts: np.ndarray, fill_bgr: Tuple[int,int,int],
                          outline_bgr: Tuple[int,int,int]=None, blur_ksize: int = 7, stroke: int = 2):
    H,W = img.shape[:2]
    mask = np.zeros((H,W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize+1
    mask_blur = cv2.GaussianBlur(mask, (k,k), 0)
    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, None]
    color_layer = np.full_like(img, fill_bgr, dtype=np.uint8)
    img[:] = (alpha * color_layer + (1.0 - alpha) * img).astype(np.uint8)
    if outline_bgr is not None:
        cv2.polylines(img, [pts], isClosed=True, color=outline_bgr, thickness=stroke, lineType=cv2.LINE_AA)

# ------------------ supersample helper ------------------
def _supersample_and_draw(img_cv: np.ndarray, draw_fn, scale: int = 2, resize_flag=cv2.INTER_LANCZOS4):
    if scale <= 1:
        draw_fn(img_cv)
        return img_cv
    H,W = img_cv.shape[:2]
    H2,W2 = H*scale, W*scale
    img_hr = cv2.resize(img_cv, (W2, H2), interpolation=cv2.INTER_CUBIC)
    draw_fn(img_hr)
    out = cv2.resize(img_hr, (W, H), interpolation=resize_flag)
    return out

# ------------------ EXTERNAL IMAGE UTILITIES ------------------
def _load_external_image(obj, target_size: Optional[Tuple[int,int]] = None, rotate: float = 0.0, flip: Optional[str] = None):
    """
    Load external_image which may be:
      - filepath string (PNG/JPG or SVG if cairosvg available)
      - PIL.Image.Image
      - np.ndarray (BGR or BGRA)
    Return: numpy array in BGRA format (H,W,4) with alpha channel (0-255).
    改动：旋转角度取负，实现「顺时针为正」（PIL/OpenCV默认逆时针，负角度=顺时针）
    """
    pil_img = None
    if isinstance(obj, str):
        ext = os.path.splitext(obj)[1].lower()
        if ext == '.svg':
            if not CAIROSVG_AVAILABLE:
                raise RuntimeError("cairosvg is required to rasterize SVG files. Install cairosvg or provide PNG/JPG.")
            png_bytes = cairosvg.svg2png(url=obj)
            if not PIL_AVAILABLE:
                raise RuntimeError("Pillow required to open rasterized SVG")
            from io import BytesIO
            pil_img = Image.open(BytesIO(png_bytes)).convert("RGBA")
        else:
            if not PIL_AVAILABLE:
                raise RuntimeError("Pillow required to load image files")
            pil_img = Image.open(obj).convert("RGBA")
    elif PIL_AVAILABLE and isinstance(obj, Image.Image):
        pil_img = obj.convert("RGBA")
    elif isinstance(obj, np.ndarray):
        arr = obj.copy()
        # 处理图像通道
        if arr.ndim == 2:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            alpha = np.full((bgr.shape[0], bgr.shape[1],1), 255, dtype=np.uint8)
            arr = np.concatenate([bgr, alpha], axis=2)
        elif arr.shape[2] == 3:
            alpha = np.full((arr.shape[0], arr.shape[1],1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=2)
        elif arr.shape[2] != 4:
            raise ValueError("Unsupported numpy image shape for external image")
        
        # 改动1：OpenCV旋转角度取负（顺时针为正）
        if target_size is not None:
            arr = cv2.resize(arr, (target_size[0], target_size[1]), interpolation=cv2.INTER_AREA)
        if rotate != 0.0:
            (h,w) = arr.shape[:2]
            # 传入 -rotate：将顺时针角度转为OpenCV默认的逆时针负角度
            M = cv2.getRotationMatrix2D((w//2, h//2), -rotate, 1.0)
            arr = cv2.warpAffine(arr, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        # 翻转逻辑不变
        if flip is not None:
            if flip in ('horizontal','both'):
                arr = cv2.flip(arr, 1)
            if flip in ('vertical','both'):
                arr = cv2.flip(arr, 0)
        return arr

    # 处理PIL图像
    if pil_img is None:
        raise RuntimeError("failed to load external image")
    
    # 调整大小
    if target_size is not None:
        tw, th = int(target_size[0]), int(target_size[1])
        pil_img = pil_img.resize((tw, th), resample=Image.LANCZOS)
    
    # 改动2：PIL旋转角度取负（顺时针为正）
    if rotate != 0.0:
        # PIL.rotate默认逆时针，传入 -rotate 实现顺时针旋转
        pil_img = pil_img.rotate(-rotate, resample=Image.BICUBIC, expand=True)
    
    # 翻转逻辑不变
    if flip is not None:
        if flip in ('horizontal','both'):
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        if flip in ('vertical','both'):
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # 转换为BGRA格式
    arr_rgba = np.array(pil_img)
    if arr_rgba.shape[2] == 4:
        arr_bgra = arr_rgba[..., [2,1,0,3]].copy()
    else:
        alpha = np.full((arr_rgba.shape[0], arr_rgba.shape[1],1), 255, dtype=np.uint8)
        arr_bgra = np.concatenate([arr_rgba[..., ::-1], alpha], axis=2)
    return arr_bgra

def _blend_overlay_alpha(canvas: np.ndarray, overlay_bgra: np.ndarray, center_xy: Tuple[int,int], opacity: float = 1.0):
    """Alpha-blend overlay onto canvas (中心对齐)"""
    Hc, Wc = canvas.shape[:2]
    Hf, Wf = overlay_bgra.shape[:2]
    cx, cy = int(center_xy[0]), int(center_xy[1])
    x0 = cx - Wf // 2
    y0 = cy - Hf // 2
    x1 = x0 + Wf
    y1 = y0 + Hf

    # 计算与画布的交集（避免越界）
    ix0 = max(0, x0); iy0 = max(0, y0)
    ix1 = min(Wc, x1); iy1 = min(Hc, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return

    # 切片处理重叠区域
    ox0 = ix0 - x0; oy0 = iy0 - y0
    ox1 = ox0 + (ix1 - ix0); oy1 = oy0 + (iy1 - iy0)
    region_canvas = canvas[iy0:iy1, ix0:ix1].astype(np.float32)
    region_overlay = overlay_bgra[oy0:oy1, ox0:ox1].astype(np.float32)

    # alpha混合计算
    over_rgb = region_overlay[..., :3]
    over_a = region_overlay[..., 3:4] / 255.0 * float(max(0.0, min(1.0, opacity)))
    out = over_rgb * over_a + region_canvas * (1.0 - over_a)
    canvas[iy0:iy1, ix0:ix1] = np.clip(out, 0, 255).astype(np.uint8)

# ------------------ Shape class ------------------
class Shape:
    def __init__(self, kind: str = 'square', size: int = 60, fill: bool = True, stroke_width: int = 2):
        supported = ('square', 'circle', 'triangle', 'diamond', 'star',
                     'pentagon', 'hexagon', 'plus', 'heart', 'crescent', 'rounded_square')
        assert kind in supported, f'unsupported shape: {kind}. supported={supported}'
        self.kind = kind
        self.size = int(size)
        self.fill = bool(fill)
        self.stroke_width = int(stroke_width)

    def draw(self, image: ImageLike, center: Tuple[int, int], angle: float = 0.0,
             color=None, outline=(0, 0, 0), flip_mode=None, **kwargs) -> np.ndarray:
        """
        绘制形状到图像上，支持wrap-around和外部图像叠加。
        关键参数：angle - 顺时针旋转角度（正角度=顺时针，负角度=逆时针）
        """
        antialias_mode = kwargs.get('antialias_mode', 'fast')
        scale = int(kwargs.get('scale', 1))
        soft_blur = int(kwargs.get('soft_blur', 7))

        # 外部图像参数
        external_obj = kwargs.get('external_image') or kwargs.get('overlay_image') or kwargs.get('texture')
        external_size = kwargs.get('external_size', None)
        external_opacity = float(kwargs.get('external_opacity', 1.0))
        external_mode = kwargs.get('external_mode', 'fit')
        external_rotate = float(kwargs.get('external_rotate', 0.0))  # 外部图像单独旋转（同样顺时针为正）
        external_flip = kwargs.get('external_flip', None)
        external_blend = kwargs.get('external_blend', 'alpha')

        # 转换输入图像为OpenCV格式
        img_cv = to_cv2(image)
        H, W = img_cv.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        s = self.size
        color_rgb = color if color is not None else rand_color()
        fill_bgr = _to_bgr_tuple(color_rgb)
        outline_bgr = _to_bgr_tuple(outline)
        half = s / 2.0

        def _rotate_points(points: List[Tuple[float, float]], angle_deg: float):
            """
            改动3：矢量形状旋转核心函数——角度取负，实现顺时针为正
            原逻辑：angle_deg → 逆时针旋转；现逻辑：-angle_deg → 顺时针旋转
            """
            a = math.radians(-angle_deg)  # 角度取负：顺时针角度 → 负的逆时针弧度
            ca, sa = math.cos(a), math.sin(a)
            return [(x * ca - y * sa, x * sa + y * ca) for (x, y) in points]

        def _apply_flip(points: List[Tuple[float, float]], flip_mode_local: str):
            """翻转逻辑不变（与旋转独立）"""
            if not flip_mode_local:
                return points
            new_pts = []
            for (x, y) in points:
                if flip_mode_local in ('horizontal', 'both'):
                    x = -x
                if flip_mode_local in ('vertical', 'both'):
                    y = -y
                new_pts.append((x, y))
            return new_pts

        def _intersects_canvas(minx, maxx, miny, maxy, Wc, Hc):
            """判断形状是否与画布重叠（用于wrap-around）"""
            if maxx < 0 or minx >= Wc:
                return False
            if maxy < 0 or miny >= Hc:
                return False
            return True

        # 内部绘制函数（支持超采样抗锯齿）
        def draw_inner(canvas, scale_local=1):
            Hc, Wc = canvas.shape[:2]
            ratio = Wc / float(W)  # 超采样比例
            cx_s = int(round(cx * ratio))
            cy_s = int(round(cy * ratio))
            s_s = int(round(s * ratio))
            stroke_s = max(1, int(round(self.stroke_width * ratio)))
            half_s = s_s / 2.0

            def pts_to_arr_abs(pts):
                """将相对坐标（以形状中心为原点）转换为画布绝对坐标"""
                return np.array([(int(round(cx_s + x)), int(round(cy_s + y))) for x, y in pts], dtype=np.int32)

            def _draw_polygon_with_offset(base_arr, dx, dy):
                """绘制带偏移的多边形（用于wrap-around）"""
                arr_shifted = base_arr + np.array([dx, dy], dtype=np.int32)
                if self.fill:
                    if antialias_mode == 'soft':
                        draw_soft_filled_poly(canvas, arr_shifted, fill_bgr, outline_bgr, blur_ksize=soft_blur, stroke=stroke_s)
                    else:
                        cv2.fillPoly(canvas, [arr_shifted], fill_bgr)
                        cv2.polylines(canvas, [arr_shifted], isClosed=True, color=outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)
                else:
                    cv2.polylines(canvas, [arr_shifted], isClosed=True, color=outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)

            def _draw_circle_with_offset(center_abs, radius, dx, dy):
                """绘制带偏移的圆形（用于wrap-around）"""
                cx_off = int(round(center_abs[0] + dx))
                cy_off = int(round(center_abs[1] + dy))
                if self.fill:
                    cv2.circle(canvas, (cx_off, cy_off), radius, fill_bgr, thickness=-1)
                    cv2.circle(canvas, (cx_off, cy_off), radius, outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(canvas, (cx_off, cy_off), radius, outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)

            # ---------------- 1. 绘制外部图像（如果提供） ----------------
            if external_obj is not None:
                try:
                    # 计算外部图像目标尺寸
                    if external_size is None:
                        target_w, target_h = int(round(s_s)), int(round(s_s))
                    elif isinstance(external_size, (list, tuple)) and len(external_size) == 2:
                        target_w, target_h = int(external_size[0]), int(external_size[1])
                    elif isinstance(external_size, (int, float)):
                        if 0 < float(external_size) <= 4.0:
                            factor = float(external_size)
                            target_w, target_h = int(round(s_s * factor)), int(round(s_s * factor))
                        else:
                            target_w, target_h = int(round(float(external_size))), int(round(float(external_size)))
                    elif isinstance(external_size, str):
                        target_w, target_h = int(round(s_s * float(external_size))), int(round(s_s * float(external_size)))
                    else:
                        target_w, target_h = int(round(s_s)), int(round(s_s))

                    # 加载外部图像（旋转已在_load_external_image中处理为顺时针）
                    overlay_bgra = _load_external_image(
                        external_obj,
                        target_size=(target_w, target_h),
                        rotate=external_rotate,  # 外部图像旋转角度（顺时针为正）
                        flip=external_flip
                    )

                    # 处理平铺模式
                    if external_mode == 'tile':
                        tile_h, tile_w = overlay_bgra.shape[:2]
                        reps_x = max(1, int(math.ceil(target_w / float(tile_w))))
                        reps_y = max(1, int(math.ceil(target_h / float(tile_h))))
                        tiled = np.tile(overlay_bgra, (reps_y, reps_x, 1))
                        overlay_bgra = tiled[:target_h, :target_w, :]

                    # wrap-around绘制外部图像
                    of_h, of_w = overlay_bgra.shape[:2]
                    minx = cx_s - of_w // 2
                    maxx = minx + of_w
                    miny = cy_s - of_h // 2
                    maxy = miny + of_h
                    for ox in (-Wc, 0, Wc):
                        for oy in (-Hc, 0, Hc):
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _blend_overlay_alpha(canvas, overlay_bgra, (cx_s + ox, cy_s + oy), opacity=external_opacity)
                except Exception as e:
                    # 外部图像加载失败时，降级绘制矢量形状
                    pass

            # 如果仅需外部图像，跳过矢量形状绘制
            if kwargs.get('external_only', False):
                return

            # ---------------- 2. 绘制矢量形状 ----------------
            if self.kind == 'square':
                # 正方形顶点（相对坐标）→ 旋转 → 绝对坐标
                corners = [(-half_s, -half_s), (half_s, -half_s), (half_s, half_s), (-half_s, half_s)]
                pts = _rotate_points(corners, angle)
                pts = _apply_flip(pts, flip_mode)
                arr_abs = pts_to_arr_abs(pts)
                xs, ys = arr_abs[:,0], arr_abs[:,1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制（画布边界外重复）
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(arr_abs, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(arr_abs, ox, oy)

            elif self.kind == 'circle':
                # 圆形旋转无视觉变化，仅处理wrap-around
                radius = max(1, int(round(half_s)))
                center_abs = (cx_s, cy_s)
                minx = center_abs[0] - radius
                maxx = center_abs[0] + radius
                miny = center_abs[1] - radius
                maxy = center_abs[1] + radius
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                            _draw_circle_with_offset(center_abs, radius, ox, oy)

            elif self.kind == 'triangle':
                # 三角形顶点（相对坐标）→ 旋转 → 绝对坐标
                pts = [(-half_s, half_s), (0, -half_s), (half_s, half_s)]
                pts = _rotate_points(pts, angle)
                pts = _apply_flip(pts, flip_mode)
                arr_abs = pts_to_arr_abs(pts)
                xs, ys = arr_abs[:,0], arr_abs[:,1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(arr_abs, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(arr_abs, ox, oy)

            elif self.kind == 'diamond':
                # 菱形顶点（相对坐标）→ 旋转 → 绝对坐标
                pts = [(0, -half_s), (half_s, 0), (0, half_s), (-half_s, 0)]
                pts = _rotate_points(pts, angle)
                pts = _apply_flip(pts, flip_mode)
                arr_abs = pts_to_arr_abs(pts)
                xs, ys = arr_abs[:,0], arr_abs[:,1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(arr_abs, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(arr_abs, ox, oy)

            elif self.kind == 'star':
                # 五角星顶点（相对坐标）→ 旋转 → 绝对坐标
                pts_list = []
                for i in range(5):
                    a_ang = math.radians(i * 72 - 90)  # 初始角度（朝上）
                    x = half_s * math.cos(a_ang)
                    y = half_s * math.sin(a_ang)
                    pts_list.append((x, y))
                pts = _rotate_points(pts_list, angle)
                pts = _apply_flip(pts, flip_mode)
                arr_abs = pts_to_arr_abs(pts)
                xs, ys = arr_abs[:,0], arr_abs[:,1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(arr_abs, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(arr_abs, ox, oy)

            elif self.kind in ('pentagon', 'hexagon'):
                # 五边形/六边形顶点（相对坐标）→ 旋转 → 绝对坐标
                n = 5 if self.kind == 'pentagon' else 6
                pts_list = []
                for i in range(n):
                    a_ang = math.radians(i * (360.0 / n) - 90)  # 初始角度（朝上）
                    x = half_s * math.cos(a_ang)
                    y = half_s * math.sin(a_ang)
                    pts_list.append((x, y))
                pts = _rotate_points(pts_list, angle)
                pts = _apply_flip(pts, flip_mode)
                arr_abs = pts_to_arr_abs(pts)
                xs, ys = arr_abs[:,0], arr_abs[:,1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(arr_abs, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(arr_abs, ox, oy)

            elif self.kind == 'plus':
                # 十字形（两个矩形）→ 旋转 → 绝对坐标
                arm = int(round(s_s * 0.25))
                length = int(round(s_s * 0.9))
                # 垂直矩形顶点
                verts = [(-arm/2, -length/2), (arm/2, -length/2), (arm/2, length/2), (-arm/2, length/2)]
                # 水平矩形顶点
                verts_h = [(-length/2, -arm/2), (length/2, -arm/2), (length/2, arm/2), (-length/2, arm/2)]
                
                # 分别旋转两个矩形
                v1 = _rotate_points(verts, angle)
                v1 = _apply_flip(v1, flip_mode)
                a1 = pts_to_arr_abs(v1)
                v2 = _rotate_points(verts_h, angle)
                v2 = _apply_flip(v2, flip_mode)
                a2 = pts_to_arr_abs(v2)

                # 计算整体边界
                xs = np.concatenate([a1[:,0], a2[:,0]])
                ys = np.concatenate([a1[:,1], a2[:,1]])
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(a1, 0, 0)
                            _draw_polygon_with_offset(a2, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(a1, ox, oy)
                                _draw_polygon_with_offset(a2, ox, oy)

            elif self.kind == 'heart':
                # 使用心形参数方程绘制更自然的爱心
                r = half_s * 0.8  # 调整缩放比例
                num_points = 60   # 增加采样点数量使曲线更平滑
                pts = []
                
                # 心形参数方程: x = 16sin³(t), y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
                for t in np.linspace(0, 2 * math.pi, num=num_points):
                    x = 16 * (math.sin(t) ** 3)
                    y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
                    
                    # 缩放并调整位置
                    scale = r / 16  # 基于参数方程的最大值进行缩放
                    x = x * scale
                    y = -y * scale  # 负号用于翻转，使心形正立
                    
                    pts.append((x, y))
                
                # 旋转+翻转+转换为绝对坐标
                pts = _rotate_points(pts, angle)
                pts = _apply_flip(pts, flip_mode)
                arr_abs = pts_to_arr_abs(pts)
                xs, ys = arr_abs[:,0], arr_abs[:,1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(arr_abs, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(arr_abs, ox, oy)

            elif self.kind == 'crescent':
                # 改动4：月牙形状旋转（内圆位置随顺时针旋转调整）
                outer_r = max(1, int(round(half_s)))
                inner_r = int(round(outer_r * 0.65))
                offset = int(round(outer_r * 0.35))  # 内圆相对外圆的初始偏移（右移）
                center_abs = (cx_s, cy_s)

                # 计算内圆相对外圆的旋转偏移（顺时针角度取负）
                a = math.radians(-angle)  # 角度取负：顺时针 → 负的逆时针弧度
                rel_x = offset  # 初始X偏移（右）
                rel_y = 0       # 初始Y偏移（无）
                # 旋转后的相对偏移
                rot_rel_x = rel_x * math.cos(a) - rel_y * math.sin(a)
                rot_rel_y = rel_x * math.sin(a) + rel_y * math.cos(a)
                inner_center_abs = (int(round(center_abs[0] + rot_rel_x)), int(round(center_abs[1] + rot_rel_y)))

                # 绘制主月牙（中心位置）
                mask = np.zeros((Hc, Wc), dtype=np.uint8)
                cv2.circle(mask, center_abs, outer_r, 255, thickness=-1)
                cv2.circle(mask, inner_center_abs, inner_r, 0, thickness=-1)
                color_layer = np.full_like(canvas, fill_bgr, dtype=np.uint8)
                alpha = (mask.astype(np.float32) / 255.0)[:, :, None]

                if self.fill:
                    canvas[:] = (alpha * color_layer + (1.0 - alpha) * canvas).astype(np.uint8)
                    # 绘制内外圆轮廓
                    cv2.circle(canvas, center_abs, outer_r, outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, inner_center_abs, inner_r, outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(canvas, center_abs, outer_r, outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, inner_center_abs, inner_r, outline_bgr, thickness=stroke_s, lineType=cv2.LINE_AA)

                # wrap-around绘制（边界外的月牙）
                minx = center_abs[0] - outer_r
                maxx = center_abs[0] + outer_r
                miny = center_abs[1] - outer_r
                maxy = center_abs[1] + outer_r
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            continue
                        if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                            mask2 = np.zeros((Hc, Wc), dtype=np.uint8)
                            outer_center_off = (center_abs[0] + ox, center_abs[1] + oy)
                            inner_center_off = (int(round(outer_center_off[0] + rot_rel_x)), int(round(outer_center_off[1] + rot_rel_y)))
                            cv2.circle(mask2, outer_center_off, outer_r, 255, thickness=-1)
                            cv2.circle(mask2, inner_center_off, inner_r, 0, thickness=-1)
                            alpha2 = (mask2.astype(np.float32) / 255.0)[:, :, None]
                            canvas[:] = (alpha2 * color_layer + (1.0 - alpha2) * canvas).astype(np.uint8)

            elif self.kind == 'rounded_square':
                # 改动5：圆角矩形（点集旋转，支持顺时针）
                r = int(round(half_s * 0.4))
                r = min(r, int(round(half_s - 1)))  # 限制圆角半径（避免超过边长）
                pts_list = []

                # 1. 计算四个圆角的相对圆心（以形状中心为原点）
                tl_cx_rel = -half_s + r  # 左上角圆心X
                tl_cy_rel = -half_s + r  # 左上角圆心Y
                tr_cx_rel = half_s - r   # 右上角圆心X
                tr_cy_rel = -half_s + r  # 右上角圆心Y
                br_cx_rel = half_s - r   # 右下角圆心X
                br_cy_rel = half_s - r   # 右下角圆心Y
                bl_cx_rel = -half_s + r  # 左下角圆心X
                bl_cy_rel = half_s - r   # 左下角圆心Y

                # 2. 生成左上角圆弧点（180°→270°，12个采样点）
                for theta in np.linspace(math.pi, 3 * math.pi / 2, num=12):
                    x = tl_cx_rel + r * math.cos(theta)
                    y = tl_cy_rel + r * math.sin(theta)
                    pts_list.append((x, y))
                # 3. 上边缘直线段（左上角圆弧终点→右上角圆弧起点）
                pts_list.append((tr_cx_rel, tr_cy_rel - r))
                # 4. 右上角圆弧点（270°→360°，12个采样点）
                for theta in np.linspace(3 * math.pi / 2, 2 * math.pi, num=12):
                    x = tr_cx_rel + r * math.cos(theta)
                    y = tr_cy_rel + r * math.sin(theta)
                    pts_list.append((x, y))
                # 5. 右边缘直线段（右上角圆弧终点→右下角圆弧起点）
                pts_list.append((br_cx_rel + r, br_cy_rel))
                # 6. 右下角圆弧点（0°→90°，12个采样点）
                for theta in np.linspace(0, math.pi / 2, num=12):
                    x = br_cx_rel + r * math.cos(theta)
                    y = br_cy_rel + r * math.sin(theta)
                    pts_list.append((x, y))
                # 7. 下边缘直线段（右下角圆弧终点→左下角圆弧起点）
                pts_list.append((bl_cx_rel, bl_cy_rel + r))
                # 8. 左下角圆弧点（90°→180°，12个采样点）
                for theta in np.linspace(math.pi / 2, math.pi, num=12):
                    x = bl_cx_rel + r * math.cos(theta)
                    y = bl_cy_rel + r * math.sin(theta)
                    pts_list.append((x, y))
                # 9. 左边缘直线段（左下角圆弧终点→左上角圆弧起点）
                pts_list.append((tl_cx_rel - r, tl_cy_rel))

                # 旋转+翻转+转换为绝对坐标（依赖_rotate_points实现顺时针）
                pts = _rotate_points(pts_list, angle)
                pts = _apply_flip(pts, flip_mode)
                arr_abs = pts_to_arr_abs(pts)
                xs, ys = arr_abs[:,0], arr_abs[:,1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # wrap-around绘制
                for ox in (-Wc, 0, Wc):
                    for oy in (-Hc, 0, Hc):
                        if ox == 0 and oy == 0:
                            _draw_polygon_with_offset(arr_abs, 0, 0)
                        else:
                            if _intersects_canvas(minx + ox, maxx + ox, miny + oy, maxy + oy, Wc, Hc):
                                _draw_polygon_with_offset(arr_abs, ox, oy)

        # ---------------- 选择抗锯齿策略 ----------------
        if antialias_mode == 'hq' and scale > 1:
            # 超采样抗锯齿（高画质）
            out = _supersample_and_draw(img_cv, draw_inner, scale=scale, resize_flag=cv2.INTER_LANCZOS4)
        else:
            # 普通绘制（快速/软抗锯齿）
            draw_inner(img_cv)
            out = img_cv

        return out