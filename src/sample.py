import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

def rand_color(min_v=30, max_v=220):
    """生成随机 BGR 颜色"""
    return tuple(int(random.uniform(min_v, max_v)) for _ in range(3))

def populate_prototype(
    W: int, 
    H: int, 
    bg_color: Tuple[int,int,int]=(255,255,255),
    seed: Optional[int] = None, 
    use_grid: bool = False,
    grid_size: int = 3, 
    cell_jitter_frac: float = 0.2,
    sample_num: Optional[int] = None,
    arrangement: Optional[str] = None  # 新增：排列方式 'random'|'horizontal'|'vertical'|'diagonal'|'circular'
) -> Dict[str, Any]:
    """
    Generate structured 'elements' list that records element properties.

    Behavior:
      - If use_grid == True:
          * Elements are placed into distinct grid cells (grid_size x grid_size).
      - If use_grid == False:
          * Elements are placed according to 'arrangement' parameter:
              - 'random': 随机位置
              - 'horizontal': 水平排列
              - 'vertical': 垂直排列
              - 'diagonal': 对角线排列
              - 'circular': 环形排列

    Returns:
        {
            'elements': [ {kind, size, fill, center, angle, bbox, stroke_width, flip}, ... ],
            'canvas_size': (W,H)
        }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32))

    elements: List[Dict[str,Any]] = []

    if arrangement == None:
        arrangement = random.choice(['random'])

    # 确定要生成的元素数量
    if sample_num is None:
        n = random.choice([1,2,3])
    else:
        n = sample_num
    # 确保至少有一个元素
    n = max(1, n)

    # 准备网格位置（如果使用网格）
    grid_positions: List[Tuple[int,int]] = []
    fixed_size = None
    
    if use_grid and grid_size >= 1:
        # 网格模式逻辑保持不变
        cell_w = float(W) / grid_size
        cell_h = float(H) / grid_size
        for r in range(grid_size):
            for c in range(grid_size):
                cx = int(round((c + 0.5) * cell_w))
                cy = int(round((r + 0.5) * cell_h))
                grid_positions.append((cx, cy))
        random.shuffle(grid_positions)

        # 确定网格模式下的固定大小
        cell_short = min(cell_w, cell_h)
        fixed_size = int(max(8, round(cell_short * 0.6)))
        fixed_size = int(max(8, min(fixed_size, min(W, H))))
        # print(fixed_size)
    else:
        # 非网格模式下，根据排列方式确定元素大小
        # 为了规则排列更美观，使用相对统一的大小
        base_size = min(W, H) // 4
        size_variation = base_size // 3
        fixed_size = [max(6, base_size + random.randint(-size_variation, size_variation)) for _ in range(n)]

    # 计算非网格模式下的元素中心位置
    centers = []
    if not use_grid:
        margin = max(fixed_size) // 2 + 10
        available_width = W - 2 * margin
        available_height = H - 2 * margin
        
        if arrangement == 'horizontal':
            # 水平排列
            if n == 1:
                # 单个元素居中
                centers.append((W // 2, H // 2))
            else:
                # 多个元素平均分布在水平线上
                spacing = available_width / (n - 1) if n > 1 else 0
                y_pos = H // 2
                for i in range(n):
                    x_pos = margin + i * spacing
                    centers.append((int(x_pos), y_pos))
                    
        elif arrangement == 'vertical':
            # 垂直排列
            if n == 1:
                centers.append((W // 2, H // 2))
            else:
                # 多个元素平均分布在垂直线上
                spacing = available_height / (n - 1) if n > 1 else 0
                x_pos = W // 2
                for i in range(n):
                    y_pos = margin + i * spacing
                    centers.append((x_pos, int(y_pos)))
                    
        elif arrangement == 'diagonal':
            # 对角线排列
            if n == 1:
                centers.append((W // 2, H // 2))
            else:
                # 从左上角到右下角的对角线
                spacing_x = available_width / (n - 1) if n > 1 else 0
                spacing_y = available_height / (n - 1) if n > 1 else 0
                for i in range(n):
                    x_pos = margin + i * spacing_x
                    y_pos = margin + i * spacing_y
                    centers.append((int(x_pos), int(y_pos)))
                    
        elif arrangement == 'circular':
            # 环形排列
            center_x, center_y = W // 2, H // 2
            # 半径为画布较小边的1/4
            radius = min(W, H) // 4
            # 均匀分布在圆周上
            for i in range(n):
                angle = 2 * np.pi * i / n
                x_pos = center_x + radius * np.cos(angle)
                y_pos = center_y + radius * np.sin(angle)
                centers.append((int(x_pos), int(y_pos)))
                
        else:  # 'random' 或其他未定义的排列方式
            # 随机位置（保持原有行为）
            for _ in range(n):
                size = fixed_size[_] if n > 1 else fixed_size[0]
                cx = random.randint(size//2 + 5, max(W - size//2 - 5, size//2 + 5))
                cy = random.randint(size//2 + 5, max(H - size//2 - 5, size//2 + 5))
                centers.append((cx, cy))

    # 生成每个元素
    for i in range(n):
        kind = random.choice(['square','circle','triangle','diamond','star','pentagon','hexagon','plus','heart','crescent','rounded_square'])

        # 尺寸选择
        if use_grid and fixed_size is not None:
            size = fixed_size
        else:
            size = fixed_size[i]

        fill = random.choice([True, True, False])
        stroke_width = random.randint(1, 3)

        # 选择中心位置
        if use_grid and grid_positions:
            # 网格模式：从网格位置中选取并添加抖动
            base_cx, base_cy = grid_positions.pop()
            cell_w = float(W) / grid_size
            cell_h = float(H) / grid_size
            short = min(cell_w, cell_h)
            jitter = cell_jitter_frac * short
            jitter_x = int(round(random.uniform(-jitter, jitter)))
            jitter_y = int(round(random.uniform(-jitter, jitter)))
            cx = int(max(0, min(W, base_cx + jitter_x)))
            cy = int(max(0, min(H, base_cy + jitter_y)))
        else:
            # 非网格模式：使用预计算的中心位置
            cx, cy = centers[i]
            # 增加少量随机抖动使排列更自然
            jitter = min(size // 4, 10)
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)
            # 确保在画布内
            cx = max(size//2 + 5, min(cx, W - size//2 - 5))
            cy = max(size//2 + 5, min(cy, H - size//2 - 5))

        # 角度选择
        if kind == 'circle':
            angle = 0.0
        else:
            angle = float(random.choice([0, 45, 90, 135, 180]))

        # 计算边界框并确保在画布内
        half = size // 2
        bx = cx - half
        by = cy - half
        bw = size
        bh = size

        # 裁剪到画布边界
        if bx < 0:
            shift = -bx
            bx = 0
            bw = max(1, bw - shift)
        if by < 0:
            shift = -by
            by = 0
            bh = max(1, bh - shift)
        if bx + bw > W:
            bw = max(1, W - bx)
        if by + bh > H:
            bh = max(1, H - by)

        element = {
            'kind': kind,
            'size': int(size),
            'fill': bool(fill),
            'stroke_width': int(stroke_width),
            'center': (int(cx), int(cy)),
            'angle': float(angle),
            'bbox': (int(bx), int(by), int(bw), int(bh)),
            'flip': {'h': False, 'v': False},
            'color': rand_color()
        }
        elements.append(element)

    state = {'elements': elements, 'canvas_size': (W, H), 'arrangement': arrangement}
    return state
