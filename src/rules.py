# rules.py (state-first version — no detect_shapes)
"""
规则（state-first 版本）：

- 不再依赖像素级的 detect_shapes。
- get_canvas_and_elements 标准化输入：支持 state-dict / elements-list / raw-image（raw-image 会回退为单元素）。
- 大部分规则优先基于 elements（位置/大小/角度）进行变换并用 _repaint_from_elements 渲染像素图。
- Overlay 类规则仍使用像素合成以保证视觉效果，但输出元素由输入 elements 启发式合成（不做轮廓检测）。
"""

import cv2
import numpy as np
import random
import math
from typing import Tuple, Union, List, Dict, Any, Optional
import copy
from config import GenConfig
from sample import populate_prototype


# ------------------ 基于元素属性的序列变换（优先使用元素状态） ------------------
def rule_translate(history_elements, rule_info=None, config=None, use_grid: bool = False, grid_size: int = 3, **kwargs):
    """
    Translate one element in history_elements[-1].

    如果 use_grid=True: 将位移解释为网格步数（dist 单位：格），默认 grid_size=3（九宫格）。
      - 在水平方向移动时改变列 index（col += dist），垂直方向改变行 index（row += dist）。
      - 保留元素在格子内的相对偏移（offset），以保持自然抖动。
    如果 use_grid=False: 行为与原先相同，dist 为像素偏移（默认取 ±(min(W,H)//3) 等）。

    返回: (A_elems, rule_info)
    """
    if config is None:
        raise ValueError("config is required and must provide canvas_size")
    W, H = config.canvas_size

    A_elems = copy.deepcopy(history_elements[-1]) if len(history_elements) > 0 else []
    rule_info = rule_info if rule_info else {}

    if not A_elems:
        rule_info.setdefault('note', 'no_elements')
        return A_elems, rule_info

    # safe idx selection
    idx = rule_info.get('idx', random.randint(0, max(0, len(A_elems)-1)))
    if idx < 0 or idx >= len(A_elems):
        idx = random.randint(0, len(A_elems)-1)
    rule_info.setdefault('idx', idx)

    # horizontal/vertical
    is_horizontal = rule_info.get('is_horizontal', random.choice([True, False]))
    rule_info.setdefault('is_horizontal', is_horizontal)

    el = A_elems[idx]

    # get bbox/center/size robustly
    cx, cy = el.get('center', (0, 0))
    bx, by, bw, bh = el.get('bbox', (0, 0, max(1, el.get('size', 10)), max(1, el.get('size', 10))))
    bw = int(max(1, bw))
    bh = int(max(1, bh))

    if use_grid:
        # read grid_size param and ensure >=1
        grid_size = int(max(1, int(grid_size)))
        rule_info['use_grid'] = True
        rule_info.setdefault('grid_size', grid_size)

        # sample dist in grid steps if not provided
        dist = rule_info.get('dist', random.choice([-2, -1, 1, 2]))
        # ensure integer steps
        try:
            dist = int(dist)
        except Exception:
            dist = int(random.choice([-2, -1, 1, 2]))
        rule_info['dist'] = dist

        # compute cell size and current cell indices
        cell_w = float(W) / grid_size
        cell_h = float(H) / grid_size

        # determine current cell col,row by floor(cx/cell_w), floor(cy/cell_h)
        col = int(min(grid_size-1, max(0, int(cx // cell_w))))
        row = int(min(grid_size-1, max(0, int(cy // cell_h))))

        # compute offset of element relative to cell center (preserve jitter)
        cell_center_x = (col + 0.5) * cell_w
        cell_center_y = (row + 0.5) * cell_h
        offset_x = float(cx) - cell_center_x
        offset_y = float(cy) - cell_center_y

        # target cell
        if is_horizontal:
            new_col = col + dist
            new_row = row
        else:
            new_col = col
            new_row = row + dist

        # clamp to grid bounds
        new_col = new_col % grid_size
        new_row = new_row % grid_size

        # compute new center keeping offset but clamp inside cell bounds
        new_cell_center_x = (new_col + 0.5) * cell_w
        new_cell_center_y = (new_row + 0.5) * cell_h

        # # limit offset so that element does not leave its cell excessively:
        # # allowed_offset_x in [-cell_w/2 + bw/2, cell_w/2 - bw/2]
        # max_off_x = max(0.0, (cell_w - bw) / 2.0)
        # max_off_y = max(0.0, (cell_h - bh) / 2.0)
        # # clamp offsets
        # offset_x = max(-max_off_x, min(max_off_x, offset_x))
        # offset_y = max(-max_off_y, min(max_off_y, offset_y))

        new_cx = int(round(new_cell_center_x))
        new_cy = int(round(new_cell_center_y))

        # ensure within canvas
        new_cx = max(0, min(W, new_cx))
        new_cy = max(0, min(H, new_cy))

        # set updated center and bbox (center-based)
        el['center'] = (int(new_cx), int(new_cy))
        new_bx = int(round(new_cx - bw / 2.0))
        new_by = int(round(new_cy - bh / 2.0))

        # clip bbox to canvas
        if new_bx < 0:
            new_bx = 0
        if new_by < 0:
            new_by = 0
        if new_bx + bw > W:
            if W - new_bx > 0:
                bw = int(max(1, W - new_bx))
            else:
                # fallback shrink and shift
                new_bx = max(0, W - bw)
                bw = int(max(1, min(bw, W)))
        if new_by + bh > H:
            if H - new_by > 0:
                bh = int(max(1, H - new_by))
            else:
                new_by = max(0, H - bh)
                bh = int(max(1, min(bh, H)))

        el['bbox'] = (int(new_bx), int(new_by), int(bw), int(bh))

    else:
        # pixel-based translation (original behavior)
        # sample dist in pixels if not provided
        dist = rule_info.get('dist', random.choice([-2, -1, 1, 2]) * (min(W, H) // 3))
        # ensure int
        try:
            dist = int(dist)
        except Exception:
            dist = int(random.choice([-2, -1, 1, 2]) * (min(W, H) // 3))
        rule_info['dist'] = dist
        rule_info['use_grid'] = False

        if is_horizontal:
            new_cx = int(cx + dist)
            new_cy = int(cy)
            el['center'] = (new_cx, new_cy)
            bx, by, bw, bh = el.get('bbox', (0, 0, bw, bh))
            new_bx = int(bx + dist)
            new_by = int(by)
        else:
            new_cx = int(cx)
            new_cy = int(cy + dist)
            el['center'] = (new_cx, new_cy)
            bx, by, bw, bh = el.get('bbox', (0, 0, bw, bh))
            new_bx = int(bx)
            new_by = int(by + dist)

        # clip bbox to canvas bounds and adjust size if needed
        if new_bx < 0:
            new_bx = 0
        if new_by < 0:
            new_by = 0
        if new_bx + bw > W:
            if W - new_bx > 0:
                bw = int(max(1, W - new_bx))
            else:
                new_bx = max(0, W - bw)
                bw = int(max(1, min(bw, W)))
        if new_by + bh > H:
            if H - new_by > 0:
                bh = int(max(1, H - new_by))
            else:
                new_by = max(0, H - bh)
                bh = int(max(1, min(bh, H)))

        el['bbox'] = (int(new_bx), int(new_by), int(bw), int(bh))

    # write back idx/is_horizontal/dist/use_grid info for reproducibility
    rule_info['idx'] = idx
    rule_info['is_horizontal'] = is_horizontal
    rule_info.setdefault('dist', dist)
    rule_info['use_grid'] = bool(use_grid)
    rule_info.setdefault('grid_size', int(grid_size) if use_grid else None)

    return A_elems, rule_info

def rule_rotate(history_elements, rule_info=None, config=None, use_grid: bool = False, angle: Optional[float]=None, **kwargs):
    """
    对 history_elements[-1] 中的某个元素进行旋转（更新 element['angle']），
    并根据旋转角度重算 bbox（轴对齐外接包围盒）。
    对不同 kind 限制允许的角度集合，并把请求角度 snap 到最近的允许角度。
    返回: (A_elems, rule_info)
    """
    if config is None:
        raise ValueError("config is required and must provide canvas_size")
    W, H = config.canvas_size

    A_elems = copy.deepcopy(history_elements[-1])
    rule_info = rule_info if rule_info else {}

    # safe idx selection
    if len(A_elems) == 0:
        return A_elems, rule_info
    
    # 避免选中circle进行旋转
    while(1):
        idx = rule_info.get('idx', random.randint(0, max(0, len(A_elems)-1)))
        el = A_elems[idx]
        kind = el.get('kind', None)
        if kind != 'circle':
            break


    # 允许角度集合（绝对角度，度数）
    allowed_angles_by_kind = {
        'circle': [],                          # 旋转无意义
        'triangle': [30, 45, 60, 90],              # 120度周期
        'square': [30, 45, 60],                    # 90度周期
        'rounded_square': [30, 45, 60],            # 90度周期
        'diamond': [30, 45, 60, 90],           # treat as 90-degree symmetry
        'star': [30, 45, 60, 90],         # 5-fold symmetry
    }

    # 获取当前角度（绝对）
    cur_angle = float(el.get('angle', 0) or 0) % 360.0
    # 选择允许角度集合（若 kind 未知，则使用较宽松策略）
    allowed_set = allowed_angles_by_kind.get(kind, [0, 45, 90, 135, 180, 225, 270, 315])

    # 如果外部传入 angle（表示增量），优先使用并写回 rule_info.requested_angle
    if angle is not None:
        rule_info['requested_angle'] = float(angle)
        requested_delta = float(angle)
    else:
        # 如果 rule_info 中已有 angle（按你的旧逻辑可能存的是 delta），使用之；否则随机采样一个增量
        if 'requested_angle' not in rule_info:
            requested_delta = random.choice(allowed_set)
            rule_info['requested_angle'] = requested_delta
        else:
            requested_delta = rule_info['requested_angle']
            

    # 确保 rule_info 包含 idx 与请求角度
    rule_info.setdefault('idx', idx)
    rule_info.setdefault('requested_angle', requested_delta)


    # 目标绝对角度（可能不是被允许的），对其取模 360
    applied_abs = (cur_angle + requested_delta) % 360.0

    # 记录决策：requested vs applied（便于审计/复现）
    rule_info['applied_angle'] = applied_abs
    rule_info['allowed_set'] = allowed_set

    # 如果是圆形，旋转不影响 bbox 或绘制（将 angle 设为 0 或保持不变）
    if kind == 'circle':
        # 对圆形保持 angle 为 0（或保持当前值，但旋转无意义）
        el['angle'] = 0.0
        # bbox 不变（保留原 bbox）
        # 如果元素没有 bbox，则用 size/center 估算（保持与原逻辑一致）
        bx, by, bw, bh = el.get('bbox', (0, 0, max(1, el.get('size', 10)), max(1, el.get('size', 10))))
        el['bbox'] = (int(bx), int(by), int(bw), int(bh))
        return A_elems, rule_info

    # 计算实际应用的角度增量（考虑最短方向）
    # delta = applied_abs - cur_angle 但我们要把 delta 映射到 [-180,180) 以便计算正负方向
    raw_delta = (applied_abs - cur_angle) % 360.0
    if raw_delta >= 180.0:
        delta = raw_delta - 360.0
    else:
        delta = raw_delta

    # 应用新的角度
    new_angle = applied_abs % 360.0
    el['angle'] = new_angle

    # 获取当前 bbox 或根据 size 估算
    bx, by, bw, bh = el.get('bbox', (0, 0, max(1, el.get('size', 10)), max(1, el.get('size', 10))))
    bw = float(max(1.0, bw))
    bh = float(max(1.0, bh))

    # 使用实际增量 delta 计算旋转后的轴对齐外接包围盒大小
    theta = math.radians(delta)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    new_bw = bw * cos_t + bh * sin_t
    new_bh = bw * sin_t + bh * cos_t

    # 以 center 为中心重新计算左上角
    cx, cy = el.get('center', (0, 0))
    new_bx = float(cx) - new_bw / 2.0
    new_by = float(cy) - new_bh / 2.0

    # 将 bbox 坐标向下取整为 int，并保证 bbox 在画布范围内（裁剪）
    new_bw_i = int(max(1, round(new_bw)))
    new_bh_i = int(max(1, round(new_bh)))
    new_bx_i = int(round(new_bx))
    new_by_i = int(round(new_by))

    # 边界裁剪：简单约束到画布范围（保持原行为）
    if new_bx_i < 0:
        new_bx_i = 0
    if new_by_i < 0:
        new_by_i = 0
    if new_bx_i + new_bw_i > W:
        shift_x = (new_bx_i + new_bw_i) - W
        new_bx_i = max(0, new_bx_i - shift_x)
        if new_bx_i + new_bw_i > W:
            new_bw_i = max(1, W - new_bx_i)
    if new_by_i + new_bh_i > H:
        shift_y = (new_by_i + new_bh_i) - H
        new_by_i = max(0, new_by_i - shift_y)
        if new_by_i + new_bh_i > H:
            new_bh_i = max(1, H - new_by_i)

    el['bbox'] = (new_bx_i, new_by_i, new_bw_i, new_bh_i)

    return A_elems, rule_info


def rule_flip(history_elements, rule_info=None, config=None, use_grid: bool = False, grid_size: int = 3, mode=None, **kwargs):
    """
    对 history_elements[-1] 中的某个元素执行镜像（flip）。
    参数:
      - history_elements: 历史元素帧列表（每帧为 elements 列表）
      - rule_info: 可选，包含 { 'idx': int, 'mode': 'horizontal'|'vertical'|'both' } 用于复现
      - config: 需要包含 canvas_size=(W,H)
      - mode: 可选，直接传入 'horizontal'|'vertical'|'both'，否则随机采样
    返回:
      A_elems, rule_info
    """
    if config is None:
        raise ValueError("config is required and must provide canvas_size")
    W, H = config.canvas_size


    # compute cell size and current cell indices
    cell_w = float(W) / grid_size
    cell_h = float(H) / grid_size
    
    A_elems = copy.deepcopy(history_elements[-1])
    rule_info = rule_info if rule_info else {}

    while(1):
        # 选择元素索引
        idx = rule_info.get('idx', random.randint(0, max(0, len(A_elems)-1)))
        cx, cy = A_elems[idx]['center']
        if cx // cell_w != grid_size // 2 or cy // cell_h != grid_size // 2:
            break
    
    # 选择模式：horizontal / vertical / both
    if mode is None:
        mode = rule_info.get('flip_mode', random.choice(['horizontal', 'vertical', 'both']))
    else:
        # 外部传入优先写回 rule_info
        rule_info['flip_mode'] = mode

    # 保证 rule_info 包含必要字段
    if not rule_info:
        rule_info['idx'] = idx
        rule_info['flip_mode'] = mode
    else:
        rule_info.setdefault('idx', idx)
        rule_info.setdefault('flip_mode', mode)

    el = A_elems[idx]

    # 当前 center, bbox, angle
    cx, cy = el.get('center', (0, 0))
    bx, by, bw, bh = el.get('bbox', (0, 0, max(1, el.get('size', 10)), max(1, el.get('size', 10))))
    bw = int(max(1, bw))
    bh = int(max(1, bh))
    cur_angle = el.get('angle', 0)

    # 初始化 flip 字段
    flip_state = el.get('flip', {'h': False, 'v': False})
    # 根据 mode 做变换
    do_h = mode in ('horizontal', 'both')
    do_v = mode in ('vertical', 'both')

    # 1) 更新 center（画布中心镜像）
    new_cx = cx
    new_cy = cy
    if do_h:
        # 水平镜像：左右翻转，x -> W - x
        new_cx = int(round(W - cx))
        flip_state['h'] = not bool(flip_state.get('h', False))
    if do_v:
        # 垂直镜像：上下翻转，y -> H - y
        new_cy = int(round(H - cy))
        flip_state['v'] = not bool(flip_state.get('v', False))

    el['center'] = (int(new_cx), int(new_cy))

    # 2) 更新 bbox：bx,by 是左上角
    new_bx = int(round(bx))
    new_by = int(round(by))
    if do_h:
        # 新的左上角 x 从右侧对称位置计算： new_bx = W - (bx + bw)
        new_bx = int(round(W - (bx + bw)))
    if do_v:
        # 新的左上角 y 从下侧对称位置计算： new_by = int(round(H - (by + bh)))
        new_by = int(round(H - (by + bh)))

    # 保证 bbox 在画布内（裁剪/调整）
    if new_bx < 0:
        new_bx = 0
    if new_by < 0:
        new_by = 0
    if new_bx + bw > W:
        # 尝试把左上角往左移使其在边界内，否则缩小宽度
        shift_x = (new_bx + bw) - W
        new_bx = max(0, new_bx - shift_x)
        if new_bx + bw > W:
            bw = max(1, W - new_bx)
    if new_by + bh > H:
        shift_y = (new_by + bh) - H
        new_by = max(0, new_by - shift_y)
        if new_by + bh > H:
            bh = max(1, H - new_by)

    el['bbox'] = (int(new_bx), int(new_by), int(bw), int(bh))
    el['flip'] = flip_state
    el['angle'] = cur_angle

    return A_elems, rule_info

def rule_transform_many(
    history_elements,
    rule_info: Optional[Dict[str, Any]] = None,
    config=None,
    use_grid: bool = False,
    grid_size: int = 3,
    indices: Optional[List[int]] = None,
    translate: Optional[Dict[str, Any]] = None,
    rotate: Optional[Dict[str, Any]] = None,
    flip: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    核心特性：
    1. 输入rule_info优先：若传入rule_info，严格沿用其中的target_indices和elem_op_map（继续操作同一元素）
    2. 生成rule_info继承：返回的rule_info包含输入的所有操作序列，仅在缺失时补充
    3. 操作一致性：对同一元素执行与输入rule_info完全相同的操作（参数不变）
    """
    if config is None:
        raise ValueError("config is required and must provide canvas_size")
    W, H = config.canvas_size

    # 1. 初始化：区分“输入rule_info”和“生成的rule_info”（生成的继承输入的所有字段）
    A_elems = copy.deepcopy(history_elements[-1]) if len(history_elements) > 0 else []
    # 生成的rule_info = 输入rule_info的深拷贝（避免修改原始输入），无输入则新建空字典
    out_rule_info = copy.deepcopy(rule_info) if rule_info is not None else {}
    # 确保“操作序列根节点”存在，避免KeyError
    out_rule_info.setdefault('transform_many', {})
    transform_seq = out_rule_info['transform_many']  # 操作序列的核心节点

    if not A_elems:
        transform_seq['note'] = 'no_elements'
        return A_elems, out_rule_info

    n = len(A_elems)  # 当前元素总数
    target_indices = []  # 最终要操作的元素索引（优先来自输入rule_info）
    elem_op_map = {}     # 最终的元素-操作映射（优先来自输入rule_info）

    # -------------------------------------------------------------------------
    # 2. 步骤1：确定目标元素（优先复用输入rule_info的target_indices，继续操作同一元素）
    # -------------------------------------------------------------------------
    # 优先级：用户传入的indices > 输入rule_info的target_indices > 随机生成
    if indices is not None:
        # 情况1：用户明确传了indices，直接用（覆盖输入rule_info的历史）
        target_indices = indices
    else:
        # 情况2：用户没传indices，优先用输入rule_info的target_indices（继续操作同一元素）
        target_indices = transform_seq.get('target_indices', None)
        if target_indices is None:
            # 情况3：无任何历史，随机生成（1~min(3, 总元素数//2)个元素）
            max_select = min(3, n)
            select_count = random.randint(1, max_select) if max_select >= 1 else 1
            target_indices = random.sample(range(n), select_count)
    
    # 标准化目标元素索引（确保合法：在0~n-1范围内，无重复）
    target_indices = [max(0, min(n-1, int(i))) for i in target_indices]
    target_indices = sorted(list(dict.fromkeys(target_indices)))
    # 更新到生成的rule_info（确保后续复用）
    transform_seq['target_indices'] = target_indices
    m = len(target_indices)  # 目标元素数量

    # -------------------------------------------------------------------------
    # 3. 步骤2：确定元素-操作映射（优先复用输入rule_info的elem_op_map，执行相同操作）
    # -------------------------------------------------------------------------
    # 先读取输入rule_info中的elem_op_map（若有）
    input_elem_op_map = transform_seq.get('elem_op_map', {})
    # 读取输入rule_info中的操作参数（平移/旋转/翻转，优先复用）
    input_trans_param = transform_seq.get('translate_param', None)
    input_rotate_param = transform_seq.get('rotate_param', None)
    input_flip_param = transform_seq.get('flip_param', None)

    # 情况1：输入rule_info有elem_op_map，且映射的元素在当前target_indices中 → 直接复用
    valid_input_op_map = {}
    for el_idx_str, op_info in input_elem_op_map.items():
        el_idx = int(el_idx_str)
        # 仅保留当前target_indices中的元素（确保操作的是同一批元素）
        if el_idx in target_indices:
            valid_input_op_map[el_idx] = op_info
    if valid_input_op_map:
        elem_op_map = valid_input_op_map
        # 从elem_op_map中提取操作参数（确保参数与输入一致）
        translate = next(iter(elem_op_map.values()))['op_param'] if 'translate' in next(iter(elem_op_map.values()))['op_type'] else None
        rotate = next(iter(elem_op_map.values()))['op_param'] if 'rotate' in next(iter(elem_op_map.values()))['op_type'] else None
        flip = next(iter(elem_op_map.values()))['op_param'] if 'flip' in next(iter(elem_op_map.values()))['op_type'] else None
    else:
        # 情况2：无输入elem_op_map，生成新的操作映射（需先确定操作参数）
        # 3.1 生成/复用操作参数（平移/旋转/翻转）
        # 平移参数：输入rule_info有则用，无则随机生成
        if translate is None:
            translate = input_trans_param if input_trans_param is not None else _gen_rand_trans_param(use_grid, grid_size, W, H)
        # 旋转参数：输入rule_info有则用，无则随机生成
        if rotate is None:
            rotate = input_rotate_param if input_rotate_param is not None else _gen_rand_rotate_param()
        # 翻转参数：输入rule_info有则用，无则随机生成
        if flip is None:
            flip = input_flip_param if input_flip_param is not None else _gen_rand_flip_param()
        
        # 3.2 筛选有效操作（基于参数）
        valid_ops = []
        if any(k in translate for k in ['dx', 'dy', 'dist']):
            valid_ops.append('translate')
        if 'angle' in rotate:
            valid_ops.append('rotate')
        if 'mode' in flip:
            valid_ops.append('flip')
        if not valid_ops:
            transform_seq['note'] = 'no_valid_operations'
            return A_elems, out_rule_info
        
        # 3.3 为每个目标元素分配操作（随机选一种有效操作，绑定参数）
        # 广播参数（适配多个元素）
        trans_params = [copy.deepcopy(translate) for _ in range(m)]
        rotate_params = [copy.deepcopy(rotate) for _ in range(m)]
        flip_params = [copy.deepcopy(flip) for _ in range(m)]
        
        for idx_i, el_idx in enumerate(target_indices):
            chosen_op = random.choice(valid_ops)
            if chosen_op == 'translate':
                elem_op_map[el_idx] = {'op_type': 'translate', 'op_param': trans_params[idx_i]}
            elif chosen_op == 'rotate':
                elem_op_map[el_idx] = {'op_type': 'rotate', 'op_param': rotate_params[idx_i]}
            elif chosen_op == 'flip':
                elem_op_map[el_idx] = {'op_type': 'flip', 'op_param': flip_params[idx_i]}
    
    # 更新到生成的rule_info（确保后续复用）
    transform_seq['elem_op_map'] = elem_op_map
    # 同步更新操作参数（确保参数与elem_op_map一致）
    transform_seq['translate_param'] = next((op['op_param'] for op in elem_op_map.values() if op['op_type'] == 'translate'), None)
    transform_seq['rotate_param'] = next((op['op_param'] for op in elem_op_map.values() if op['op_type'] == 'rotate'), None)
    transform_seq['flip_param'] = next((op['op_param'] for op in elem_op_map.values() if op['op_type'] == 'flip'), None)
    transform_seq['valid_ops'] = [op['op_type'] for op in elem_op_map.values()]  # 记录当前有效操作类型

    # -------------------------------------------------------------------------
    # 4. 步骤3：执行操作（严格按elem_op_map，对同一元素执行相同操作）
    # -------------------------------------------------------------------------
    for el_idx, op_info in elem_op_map.items():
        el = A_elems[el_idx]  # 锁定目标元素
        op_type = op_info['op_type']
        op_param = op_info['op_param']  # 复用输入的操作参数
        kind = el.get('kind', 'default')
        size = int(el.get('size', 10))
        # 获取元素当前状态（不依赖历史，仅基于当前元素）
        cx, cy = el.get('center', (W//2, H//2))
        bx, by, bw, bh = el.get('bbox', (int(cx - size/2), int(cy - size/2), int(size), int(size)))
        bw = int(max(1, bw)); bh = int(max(1, bh))

        # 4.1 执行平移（用elem_op_map中的参数）
        if op_type == 'translate':
            use_grid = op_param.get('use_grid', False)
            mode = op_param.get('mode', 'relative')
            dx, dy = 0, 0

            # 解析平移参数（网格/像素模式）
            if 'dx' in op_param and 'dy' in op_param:
                dx = op_param['dx']
                dy = op_param['dy']
            else:
                dist = op_param.get('dist', 0)
                dir = op_param.get('dir', 'horizontal')
                dx = dist if dir == 'horizontal' else 0
                dy = dist if dir == 'vertical' else 0

            # 计算新位置（基于当前元素位置 + 固定参数）
            if use_grid:
                grid_int = max(1, op_param.get('grid_size', 3))
                cell_w = W / grid_int
                cell_h = H / grid_int
                col = min(grid_int - 1, max(0, int(cx // cell_w)))
                row = min(grid_int - 1, max(0, int(cy // cell_h)))
                new_col = (col + dx + grid_int) % grid_int
                new_row = (row + dy + grid_int) % grid_int
                cell_cx = (new_col + 0.5) * cell_w
                cell_cy = (new_row + 0.5) * cell_h
                offset_x = cx - (col + 0.5) * cell_w
                offset_y = cy - (row + 0.5) * cell_h
                max_off_x = max(0.0, (cell_w - bw) / 2)
                max_off_y = max(0.0, (cell_h - bh) / 2)
                offset_x = max(-max_off_x, min(max_off_x, offset_x))
                offset_y = max(-max_off_y, min(max_off_y, offset_y))
                new_cx = int(round(cell_cx + offset_x))
                new_cy = int(round(cell_cy + offset_y))
            else:
                new_cx = int(round(cx + dx))
                new_cy = int(round(cy + dy))

            # 边界裁剪（确保不超出画布）
            new_cx = (new_cx + W) % W
            new_cy = (new_cy + H) % H
            # 更新元素状态
            el['center'] = (new_cx, new_cy)
            new_bx = int(round(new_cx - bw / 2))
            new_by = int(round(new_cy - bh / 2))
            new_bx = max(0, new_bx)
            new_by = max(0, new_by)
            if new_bx + bw > W:
                bw = int(max(1, W - new_bx))
            if new_by + bh > H:
                bh = int(max(1, H - new_by))
            el['bbox'] = (new_bx, new_by, bw, bh)

        # 4.2 执行旋转（用elem_op_map中的参数）
        elif op_type == 'rotate':
            angle_delta = op_param.get('angle', 0)  # 固定角度增量
            snap_map = op_param.get('snap_map', {'default': [45, 90, 135, 180, 225, 270, 315]})
            # 角度快照（基于元素类型 + 固定参数）
            allowed_angles = snap_map.get(kind, snap_map['default'])
            if angle_delta != 0 and allowed_angles:
                angle_delta = random.choice(allowed_angles)
            op_param['angle'] = angle_delta

            cur_angle = float(el.get('angle', 0))
            new_angle = (cur_angle + angle_delta) % 360.0

            
            # 更新元素状态
            el['angle'] = new_angle
            delta_rad = math.radians(new_angle - cur_angle)
            cos_t = abs(math.cos(delta_rad))
            sin_t = abs(math.sin(delta_rad))
            new_bw = int(max(1, round(bw * cos_t + bh * sin_t)))
            new_bh = int(max(1, round(bw * sin_t + bh * cos_t)))
            new_bx = int(round(cx - new_bw / 2))
            new_by = int(round(cy - new_bh / 2))
            new_bx = max(0, new_bx)
            new_by = max(0, new_by)
            if new_bx + new_bw > W:
                new_bw = int(max(1, W - new_bx))
            if new_by + new_bh > H:
                new_bh = int(max(1, H - new_by))
            el['bbox'] = (new_bx, new_by, new_bw, new_bh)

        # 4.3 执行翻转（用elem_op_map中的参数）
        elif op_type == 'flip':
            mode = op_param.get('mode', 'horizontal')  # 固定翻转模式
            do_h = mode in ('horizontal', 'both')
            do_v = mode in ('vertical', 'both')

            # 计算新状态（基于当前元素位置 + 固定参数）
            new_cx = cx if not do_h else int(round(W - cx))
            new_cy = cy if not do_v else int(round(H - cy))
            new_bx = bx if not do_h else int(round(W - (bx + bw)))
            new_by = by if not do_v else int(round(H - (by + bh)))

            # 边界裁剪
            new_bx = max(0, new_bx)
            new_by = max(0, new_by)
            if new_bx + bw > W:
                bw = int(max(1, W - new_bx))
            if new_by + bh > H:
                bh = int(max(1, H - new_by))

            # 更新元素状态
            el['center'] = (new_cx, new_cy)
            el['bbox'] = (new_bx, new_by, bw, bh)
            # 更新翻转状态（切换）
            flip_state = el.get('flip', {'h': False, 'v': False})
            flip_state['h'] = not flip_state['h'] if do_h else flip_state['h']
            flip_state['v'] = not flip_state['v'] if do_v else flip_state['v']
            el['flip'] = flip_state

    # -------------------------------------------------------------------------
    # 5. 返回结果：操作后的元素 + 继承/补充后的rule_info
    # -------------------------------------------------------------------------
    return A_elems, out_rule_info


# ------------------------------ 辅助函数（生成随机参数，仅在无输入时用）------------------------------
def _gen_rand_trans_param(use_grid: bool, grid_size: int, W: int, H: int) -> Dict:
    """生成随机平移参数（仅在无输入rule_info时用）"""
    if use_grid:
        return {
            'dist': random.choice([-2, -1, 1, 2]),
            'dir': random.choice(['horizontal', 'vertical']),
            'mode': 'relative',
            'use_grid': use_grid,
            'grid_size': grid_size
        }
    else:
        max_off = min(W, H) // 6
        min_off = min(W, H) // 10
        off = random.randint(min_off, max_off)
        dx = random.choice([-off, off]) if random.random() < 0.5 else 0
        dy = -off if dx == 0 else 0
        return {'dx': dx, 'dy': dy, 'mode': 'relative', 'use_grid': use_grid}


def _gen_rand_rotate_param() -> Dict:
    """生成随机旋转参数（仅在无输入rule_info时用）"""
    return {
        'angle': random.choice([45, 90, 135, 180, 225, 270, 315]),
        'snap_map': {
            'circle': [],                          # 旋转无意义
            'triangle': [30, 45, 60, 90],              # 120度周期
            'square': [30, 45, 60],                    # 90度周期
            'rounded_square': [30, 45, 60],            # 90度周期
            'diamond': [30, 45, 60, 90],           # treat as 90-degree symmetry
            'star': [30, 45, 60, 90],         # 5-fold symmetry
            'default': [0, 45, 90, 135, 180, 225, 270, 315]
        }
    }


def _gen_rand_flip_param() -> Dict:
    """生成随机翻转参数（仅在无输入rule_info时用）"""
    return {'mode': random.choice(['horizontal', 'vertical', 'both'])}


def rule_traverse_sequence(
    history_elements: List[List[Dict[str, Any]]],
    rule_info: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    kinds: Optional[List[str]] = None,
    use_grid: bool = False,
    seq_len: int = 3,
    placement: str = 'stack_right',  # 'stack_right' | 'random' | 'grid'
    size_hint: int = 80,             # 基准尺寸（像素）用于 bbox/visual size
    grid_cols: int = 3,
    grid_rows: int = 3,
    **kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    单一遍历规则：按序列遍历修改现有元素的kind属性（不再叠加新元素）。
    - 与原版本的主要区别：不添加新元素，而是替换现有元素的kind属性
    - history_elements: 历史状态列表，每项通常是当前画布的元素列表；要求 history_elements[-1] 存在。
    - rule_info: 用于保存/恢复序列、当前索引等（会被修改并返回）。
      keys used:
        - 'sequence': list of kinds (strings)
        - 'step_idx': 当前要修改的索引
        - 'done': True 表示序列已遍历完
    - config: 需要提供 canvas_size 属性 (W, H)。若为 None，则使用默认 800x600。
    - kinds: 允许的形状种类（默认 ['square','circle','triangle','diamond','star']）
    - seq_len: 序列总长度（>=2）
    - placement: 元素位置调整策略（'stack_right' 沿 X 轴堆叠，'random' 随机，'grid' 采用 grid 放置）
    - 返回: (A_elems, rule_info)
    """

    if rule_info is None:
        rule_info = {}

    if kinds is None:
        kinds = ['square', 'circle', 'triangle', 'diamond', 'star']

    # canvas size
    if config is None or not hasattr(config, 'canvas_size'):
        W, H = 512, 512
    else:
        W, H = config.canvas_size

    # get current canvas elements (deep copy)
    A_elems = copy.deepcopy(history_elements[-1]) if len(history_elements) > 0 else []
    if not A_elems:
        rule_info.setdefault('note', 'no_elements_in_history')
        return A_elems, rule_info

    # 使用第一个元素作为序列锚点（而非最后一个）
    first_el = A_elems[0]
    second_el = A_elems[1]
    anchor_kinds = [first_el.get('kind'), second_el.get('kind')]
    # if all(anchor_kind not in kinds for anchor_kind in anchor_kinds):
    #     # 如果锚点类型不在允许列表中，随机选择一个并设置
    #     anchor_kind = random.choice(kinds)
    #     first_el['kind'] = anchor_kind
    #     anchor_kind = random.choice(kinds)
    #     second_el['kind'] = anchor_kind
    #     anchor_kinds = [first_el.get('kind'), second_el.get('kind')]

    # 准备序列（如果不存在）
    sequence = rule_info.get('sequence')
    if sequence is None:
        # 生成随机序列，确保第一个元素与锚点一致
        seq_len = max(2, int(max(2, seq_len)))
        sequence = anchor_kinds
        # 填充剩余元素
        for _ in range(seq_len - 2):
            sequence.append(random.choice(kinds))
        rule_info['sequence'] = sequence
        rule_info['step_idx'] = 1
        rule_info['done'] = False

    # 读取状态
    step_idx = int(rule_info.get('step_idx', 1))
    if rule_info.get('done', False):
        rule_info.setdefault('note', 'sequence_already_done')
        return A_elems, rule_info

    # 检查序列是否过短或已完成
    if not isinstance(sequence, list) or len(sequence) < 2 or step_idx-1 >= len(sequence):
        rule_info['done'] = True
        rule_info.setdefault('note', 'sequence_finished')
        return A_elems, rule_info

    rule_info.setdefault('last_modified', [])
    
    for elem_idx in range(2):
        # 确定当前步骤要设置的元素类型
        step_idx = step_idx % len(sequence)
        current_kind = sequence[step_idx]

        target_elem = A_elems[elem_idx]

        # 保存原始类型用于记录
        original_kind = target_elem.get('kind', 'unknown')
        
        # 替换元素的kind属性
        target_elem['kind'] = current_kind
        # target_elem['size'] = int(size_hint)  # 更新尺寸
        
        # 记录修改信息
        target_elem.setdefault('meta', {})
        target_elem['meta'].update({
            'step_idx': step_idx,
            'original_kind': original_kind,
            'sequence': sequence.copy()
        })

        # 更新视觉信息
        target_elem.setdefault('visual', {})
        target_elem['visual']['type'] = current_kind

        # 标记完成状态
        if rule_info['step_idx'] >= len(sequence):
            rule_info['done'] = True
            rule_info.setdefault('note', 'sequence_finished_after_this_step')
        else:
            rule_info['done'] = False

        # 记录最后修改的元素信息
        rule_info['last_modified'].append({
            'element_index': elem_idx,
            'from_kind': original_kind,
            'to_kind': current_kind,
            'center': target_elem['center'],
            'bbox': target_elem['bbox']
        })
        step_idx += 1

    # 推进步骤索引
    rule_info['step_idx'] = step_idx-1

    return A_elems, rule_info


def rule_traverse_positions(
    history_elements: List[List[Dict[str, Any]]],
    rule_info: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    placement: str = 'random',  # 'grid' | 'stack_right' | 'random' 位置生成策略
    seq_len: int = 3,         # 位置序列总长度（需≥2，每次取2个位置）
    size_hint: int = 80,      # 元素尺寸（用于计算边界框和位置间隔）
    grid_cols: int = 3,       # 网格列数（placement='grid'时生效）
    grid_rows: int = 3,       # 网格行数（placement='grid'时生效）
    **kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    位置遍历规则：当history_elements最后一个元素列表仅有1个元素时，生成位置序列，
    每次从序列中取两个位置替换该元素（生成两个同属性不同位置的元素）。
    
    - history_elements: 历史状态列表，要求 history_elements[-1] 存在且长度为1。
    - rule_info: 用于保存/恢复位置序列、当前步骤等状态（会被修改并返回）。
      包含键：
        - 'positions_sequence': 位置序列（每个元素为中心坐标 (x, y)）
        - 'step_idx': 当前步骤索引（每次取2个位置，步长为1）
        - 'done': 是否完成遍历
    - config: 需提供 canvas_size 属性 (W, H)，若为None则默认512x512。
    - placement: 位置生成策略（网格/右堆叠/随机）
    - seq_len: 位置序列总长度（需≥2，建议为偶数）
    - 返回: (A_elems, rule_info) 新元素列表和更新后的状态
    """
    if rule_info is None:
        rule_info = {}
    
    
    # 获取画布尺寸
    if config is None or not hasattr(config, 'canvas_size'):
        W, H = 512, 512  # 默认画布大小
    else:
        W, H = config.canvas_size
    
    # 深拷贝当前元素
    current_elems = copy.deepcopy(history_elements[-1])
    A_elems = []  # 用于存储新元素（最终长度为2）
    
    # 生成位置序列（若未初始化）
    positions_sequence = rule_info.get('positions_sequence')
    if positions_sequence is None:
        # 确保序列长度至少为2，且建议为偶数
        seq_len = max(2, seq_len)
        positions_sequence = [history_elements[0][0]['center'], history_elements[0][1]['center']]
        # 根据placement策略生成位置
        if placement == 'grid':
            # 网格布局：计算网格中每个位置的中心坐标
            total_grids = grid_cols * grid_rows
            step_x = W / (grid_cols + 1)  # 列间隔
            step_y = H / (grid_rows + 1)  # 行间隔
            for row in range(grid_rows):
                for col in range(grid_cols):
                    x = step_x * (col + 1)
                    y = step_y * (row + 1)
                    positions_sequence.append((x, y))
                    if len(positions_sequence) >= seq_len:
                        break
                if len(positions_sequence) >= seq_len:
                    break
        
        elif placement == 'stack_right':
            # 右堆叠布局：沿X轴依次排列，Y轴居中
            start_x = size_hint * 1.5  # 起始X坐标（避免贴边）
            start_y = H / 2  # Y轴居中
            step = size_hint * 1.8  # 元素间隔（避免重叠）
            for i in range(seq_len-2):
                x = start_x + i * step
                # 确保不超出画布右边界
                if x + size_hint/2 > W:
                    x = W - size_hint/2
                positions_sequence.append((x, start_y))
        
        else:  # 'random' 随机布局
            # 随机位置（确保元素完全在画布内）
            min_coord = size_hint / 2
            max_x = W - min_coord
            max_y = H - min_coord
            for _ in range(seq_len-2):
                x = random.uniform(min_coord, max_x)
                y = random.uniform(min_coord, max_y)
                positions_sequence.append((x, y))
        
        # 保存序列并初始化状态
        rule_info['positions_sequence'] = positions_sequence
        rule_info['step_idx'] = 1
        rule_info['done'] = False
    
    # 读取当前状态
    step_idx = int(rule_info.get('step_idx', 1))
    positions = rule_info['positions_sequence']
    if rule_info.get('done', False):
        rule_info.setdefault('note', 'positions_sequence_already_done')
        return (history_elements[-1], rule_info)  # 已完成则返回当前元素
    
    # 检查序列是否足够（每次需要2个位置）
    if len(positions) < 2 or (step_idx-1) >= len(positions):
        rule_info['done'] = True
        rule_info.setdefault('note', 'positions_sequence_finished')
        return (history_elements[-1], rule_info)
    
    # 从序列中取当前步骤的两个位置
    pos1 = positions[step_idx % len(positions)]
    pos2 = positions[(step_idx + 1) % len(positions)]
    
    # 生成两个新元素（复制原元素属性，仅修改位置）
    for idx, (x, y) in enumerate([pos1, pos2]):
        new_elem = current_elems[idx]
        # 更新中心坐标
        new_elem['center'] = (x, y)
        # 更新边界框（基于中心和尺寸）
        s = size_hint
        new_elem['bbox'] = (x - s/2, y - s/2, x + s/2, y + s/2)
    A_elems = current_elems
    
    # 更新状态信息
    rule_info.setdefault('last_modified', [])
    rule_info['last_modified'].append({
        'step_idx': step_idx,
        'original_center': [current_elems[0]['center'], current_elems[1]['center']],
        'new_centers': [pos1, pos2],
        'elements_count': 2
    })
    
    # 推进步骤（下次取后面的两个位置）
    next_step = step_idx + 1
    rule_info['step_idx'] = next_step
    # 检查是否完成（下次步骤是否超出序列）
    if (next_step-2) >= len(positions):
        rule_info.setdefault('note', 'positions_sequence_will_finish_next_step')
    
    return A_elems, rule_info

def rule_element_transfer(A, B, **kwargs):
    A_img, A_elems, A_size = get_canvas_and_elements(A)
    B_img, B_elems, B_size = get_canvas_and_elements(B)
    # choose biggest element from A and paste its bbox content into center of B (pixel-level)
    elems_sorted = sorted(A_elems, key=lambda e: e.get('size', e.get('area',0)), reverse=True)
    if not elems_sorted:
        return {'img': B_img.copy(), 'elements': B_elems, 'canvas_size': (B_size[0], B_size[1])}
    chosen = elems_sorted[0]
    x,y,w,h = chosen.get('bbox', (0,0, max(1, chosen.get('size',10)), max(1, chosen.get('size',10))))
    x = max(0,x); y = max(0,y)
    h = max(1,h); w = max(1,w)
    crop = A_img[y:y+h, x:x+w].copy()
    out = B_img.copy()
    paste(crop, out, (max(0, out.shape[1]//2 - w//2), max(0, out.shape[0]//2 - h//2)))
    # update B_elems by adding a transferred element at center
    new_el = {
        'kind': chosen.get('kind','unknown'),
        'center': (out.shape[1]//2, out.shape[0]//2),
        'bbox': (out.shape[1]//2 - w//2, out.shape[0]//2 - h//2, w, h),
        'size': chosen.get('size', max(w,h)),
        'fill': chosen.get('fill', True),
        'angle': chosen.get('angle', None),
        'stroke_width': chosen.get('stroke_width', 1)
    }
    new_elements = list(B_elems) + [new_el]
    return {'img': out, 'elements': new_elements, 'canvas_size': (out.shape[1], out.shape[0])}

# ------------------ overlay wrapper rules（接受 state dicts / elements / image） ------------------
def _merge_elements_for_direct(A_elems, B_elems):
    # naive merge: keep both lists, try to adjust bboxes if overlapping
    merged = []
    merged.extend([dict(el) for el in A_elems])
    merged.extend([dict(el) for el in B_elems])
    return merged

def _elements_for_diff_keep_same(A_img, A_elems, B_img, B_elems, out_img):
    # heuristic: prefer elements from A that remain visually in out_img (we use bbox overlap)
    res = []
    H,W = out_img.shape[:2]
    out_mask = (cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY) < 250).astype(np.uint8)
    for el in A_elems:
        bx,by,bw,bh = el.get('bbox',(0,0,0,0))
        # clip bbox
        x0 = max(0,bx); y0 = max(0,by); x1 = min(W, bx+bw); y1 = min(H, by+bh)
        if x1 > x0 and y1 > y0:
            region = out_mask[y0:y1, x0:x1]
            if region.sum() > 0:
                res.append(dict(el))
    return res

def _elements_for_diff_keep_intersection(A_img, A_elems, B_img, B_elems, out_img):
    # heuristic: keep elements whose bbox area still present in out_img and also similar in A
    res = []
    H,W = out_img.shape[:2]
    out_mask = (cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY) < 250).astype(np.uint8)
    for el in A_elems:
        bx,by,bw,bh = el.get('bbox',(0,0,0,0))
        x0 = max(0,bx); y0 = max(0,by); x1 = min(W, bx+bw); y1 = min(H, by+bh)
        if x1 > x0 and y1 > y0:
            region = out_mask[y0:y1, x0:x1]
            if region.sum() > (bw*bh)//8:  # some portion remains
                res.append(dict(el))
    return res

def rule_direct_overlay(history_elements, rule_info=None, config=None, use_grid: bool = False, grid_size: int = 3, **kwargs):
    """
    规则：直接叠加（overlay）元素（已修改）：
    - 若 len(history_elements) % 3 != 2:
        替换最后一帧 elements 中若干个元素（不能全部替换）。
        如果最后一帧元素数 <= 1，则尝试生成并加入一个元素（fallback 为简单随机元素）。
    - 若 len(history_elements) % 3 == 2:
        对最后两个帧进行“叠加合并”，删除重复部分（在 metadata 中记录 overlap bbox）。
    参数:
    - history_elements: list of elements-lists (每帧为 elements 列表)
    - rule_info: 用于可复现采样的字典（会被写入 idx / op / seed / overlap_bbox / replaced_idx / added_idx 等）
    - config: 需要包含 canvas_size=(W,H)
    - kwargs:
      - generator 或 self: 可选，包含 populate_prototype(W,H,bg_color=(...), seed=None) 方法
      - bg_color: RGB tuple，用于可能的 populate_prototype 调用（默认白）
      - seed: 可选随机种子，回写到 rule_info
    返回:
    A_elems, rule_info
    """
    if config is None:
        raise ValueError("config is required and must provide canvas_size")
    W, H = config.canvas_size

    A_elems = copy.deepcopy(history_elements[-1])
    rule_info = rule_info if rule_info else {}

    # preserve/accept seed if provided
    seed = kwargs.get('seed', rule_info.get('seed', None))
    if seed is not None:
        random.seed(seed)
    rule_info['seed'] = seed

    # helper: try to call populate_prototype on a provided generator/self
    def _try_populate_prototype():
        bg_color = kwargs.get('bg_color', (255, 255, 255))
        # try several possible signatures, be tolerant
        try:
            # try signature: populate_prototype(self, W, H, bg_color=..., seed=...)
            res = populate_prototype(W, H, bg_color=bg_color, seed=seed, use_grid=use_grid, grid_size=grid_size)
            return res
        except TypeError:
            try:
                # try signature: populate_prototype(self, W, H, seed=None)
                res = populate_prototype(W, H, seed=seed, use_grid=use_grid, grid_size=grid_size)
                return res
            except Exception:
                try:
                    # maybe it requires an image param _populate_prototype(self, im, seed)
                    # we cannot construct an image reliably here, so skip
                    return None
                except Exception:
                    return None
        except Exception:
            return None

    # helper to convert a proto return into a list of element dicts (possibly single)
    def _elems_from_proto(proto):
        if proto is None:
            return []
        if isinstance(proto, dict):
            if 'elements' in proto and isinstance(proto['elements'], (list, tuple)):
                return [copy.deepcopy(e) for e in proto['elements']]
            else:
                return [copy.deepcopy(proto)]
        if isinstance(proto, (list, tuple)):
            return [copy.deepcopy(e) for e in proto]
        return []

    # helper: compute bbox
    def _get_bbox(el):
        bb = el.get('bbox', None)
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            bx, by, bw, bh = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            return bx, by, bw, bh
        # try derive from center and size
        cx, cy = el.get('center', (0, 0))
        s = el.get('size', max(1, int(min(W, H)//10)))
        bw = int(max(1, s))
        bh = int(max(1, s))
        bx = int(cx - bw/2)
        by = int(cy - bh/2)
        return bx, by, bw, bh

    frame_count = len(history_elements)

    # ------------------ 新规则分支 ------------------
    if frame_count % 3 != 2:
        # 随机生成k个元素

        rule_info['op'] = 'added_element_from_proto'
        rule_info.setdefault('frame_count', frame_count)

        proto = _try_populate_prototype()
        new_elem = None
        if proto:
            elems_from_proto = _elems_from_proto(proto)
        A_elems = elems_from_proto

        rule_info.setdefault('seed', seed)
        return A_elems, rule_info

    # ------------------ 保持原 overlay/merge 行为（当 frame_count %3 == 2） ------------------

    A_elems = history_elements[-1] + history_elements[-2]

    rule_info['op'] = 'merge_last_two'
    rule_info.setdefault('seed', seed)
    return A_elems, rule_info

def rule_diff_keep_same(history_elements, rule_info=None, config=None, iou_thresh: float = 0.5,
                        size_rel_thresh: float = 0.2, angle_thresh_deg: float = 5.0, use_grid: bool = False, grid_size: int = 3, **kwargs):
    """
    去同存异（diff：keep different）— 已修改：
      - 若 len(history_elements) % 3 != 2:
          替换最后一帧 elements 中若干个元素（不能全部替换）。
          如果最后一帧元素数 <= 1，则尝试通过 generator.populate_prototype(...) 生成并加入一个元素（fallback 为随机元素）。
      - 若 len(history_elements) % 3 == 2:
          比较 history_elements[-2] 与 history_elements[-1]，把判为“相同”的元素从最新帧移除。
    返回:
        A_elems, rule_info
    rule_info 包含（根据分支不同）:
        - op: 操作名
        - replaced_idx / num_replaced / added_idx / fallback_created_count（替换分支）
        - removed_idx_in_last / kept_idx_in_last（diff_keep 分支）
        - seed: 随机种子（若提供或采样）
    """
    if config is not None and hasattr(config, 'canvas_size'):
        W, H = config.canvas_size
    else:
        W, H = (200, 200)

    A_elems = copy.deepcopy(history_elements[-1]) if len(history_elements) >= 1 else []
    rule_info = rule_info if rule_info else {}

    # preserve seed if provided
    seed = kwargs.get('seed', rule_info.get('seed', None))
    if seed is not None:
        random.seed(seed)
        rule_info['seed'] = seed

    # tolerant populate_prototype caller
    def _try_populate_prototype():
        bg_color = kwargs.get('bg_color', (255, 255, 255))
        try:
            return populate_prototype(W, H, bg_color=bg_color, seed=seed, use_grid=use_grid, grid_size=grid_size)
        except TypeError:
            try:
                return populate_prototype(W, H, seed=seed, use_grid=use_grid, grid_size=grid_size)
            except Exception:
                try:
                    return None
                except Exception:
                    return None
        except Exception as e:
            return None

    def _elems_from_proto(proto):
        if proto is None:
            return []
        if isinstance(proto, dict):
            if 'elements' in proto and isinstance(proto['elements'], (list, tuple)):
                return [copy.deepcopy(e) for e in proto['elements']]
            else:
                return [copy.deepcopy(proto)]
        if isinstance(proto, (list, tuple)):
            return [copy.deepcopy(e) for e in proto]
        return []

    def _create_fallback_elem():
        kind = random.choice(['square', 'circle', 'triangle', 'diamond', 'star'])
        size = random.randint(max(8, min(W, H)//12), max(12, min(W, H)//4))
        cx = random.randint(size//2, max(size//2, W - size//2))
        cy = random.randint(size//2, max(size//2, H - size//2))
        return {
            'kind': kind,
            'size': size,
            'fill': True,
            'center': (cx, cy),
            'angle': random.choice([0, 0, 45, 90]),
            'bbox': (cx - size//2, cy - size//2, size, size),
            'color': kwargs.get('color', None)
        }

    def _get_bbox(el):
        bb = el.get('bbox', None)
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            bx, by, bw, bh = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            return bx, by, bw, bh
        cx, cy = el.get('center', (0, 0))
        s = el.get('size', None)
        if s is None:
            s = max(1, int(min(W, H) // 10))
        bw = int(max(1, s))
        bh = int(max(1, s))
        bx = int(round(cx - bw / 2.0))
        by = int(round(cy - bh / 2.0))
        return bx, by, bw, bh

    def _iou(b1, b2):
        bx1, by1, bw1, bh1 = b1
        bx2, by2, bw2, bh2 = b2
        x1 = max(bx1, bx2)
        y1 = max(by1, by2)
        x2 = min(bx1 + bw1, bx2 + bw2)
        y2 = min(by1 + bh1, by2 + bh2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = bw1 * bh1
        area2 = bw2 * bh2
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    frame_count = len(history_elements)

    # ---------- 新逻辑：当 frame_count % 3 != 2 时，替换最后一帧中若干元素 ----------
    if frame_count % 3 != 2:
        rule_info['op'] = 'replace_some_in_last_frame'
        rule_info.setdefault('frame_count', frame_count)

        n = len(A_elems)
        # 若最后一帧元素太少（<=1），尝试生成并加入一个元素（保持稳定性）
        if n <= 1:
            proto = _try_populate_prototype()
            new_elem = None
            if proto:
                elems_from_proto = _elems_from_proto(proto)
                if len(elems_from_proto) > 0:
                    new_elem = copy.deepcopy(elems_from_proto[0])
            if new_elem is None:
                new_elem = _create_fallback_elem()
                rule_info['fallback_created'] = True
            A_elems.append(new_elem)
            rule_info['added_idx'] = len(A_elems) - 1
            rule_info.setdefault('seed', seed)
            return A_elems, rule_info

        # 正常：随机选择 r ∈ [1, n-1] 个索引替换（不能全部替换）
        r = random.randint(1, max(1, n - 1))
        replaced_indices = random.sample(range(0, n), r)
        replaced_indices.sort()
        rule_info['num_replaced'] = r
        rule_info['replaced_idx'] = replaced_indices

        # 获取 r 个新元素，优先使用 populate_prototype 返回的元素
        new_elems = []
        proto = _try_populate_prototype()
        if proto:
            elems_from_proto = _elems_from_proto(proto)
            for e in elems_from_proto:
                if len(new_elems) >= r:
                    break
                new_elems.append(copy.deepcopy(e))
        # 补足不足的新元素
        while len(new_elems) < r:
            new_elems.append(_create_fallback_elem())
            rule_info.setdefault('fallback_created_count', 0)
            rule_info['fallback_created_count'] += 1

        # 在相同索引位置替换
        for i, idx in enumerate(replaced_indices):
            A_elems[idx] = new_elems[i]

        rule_info.setdefault('seed', seed)
        return A_elems, rule_info

    # ---------- 保持原 diff_keep 行为（当 frame_count % 3 == 2） ----------
    # 需要至少两帧来比较
    if len(history_elements) < 2:
        rule_info['op'] = 'diff_keep_skipped_not_enough_history'
        return A_elems, rule_info

    prev_elems = history_elements[-2]
    last_elems = history_elements[-1]

    removed_indices = []
    kept_indices = []

    # 过滤出保留的元素
    new_A = []

    for i, el_last in enumerate(last_elems):
        considered_same = False
        bx_l = _get_bbox(el_last)
        kind_l = el_last.get('kind', None)
        size_l = float(el_last.get('size', 0) or 0)
        angle_l = float(el_last.get('angle', 0) or 0)

        for j, el_prev in enumerate(prev_elems):
            kind_p = el_prev.get('kind', None)
            if kind_p != kind_l:
                continue
            bx_p = _get_bbox(el_prev)
            iou_val = _iou(bx_l, bx_p)
            size_p = float(el_prev.get('size', 0) or 0)
            size_rel_diff = 0.0
            if max(size_p, size_l) > 0:
                size_rel_diff = abs(size_p - size_l) / max(size_p, size_l)
            angle_p = float(el_prev.get('angle', 0) or 0)
            angle_diff = abs(((angle_l - angle_p + 180) % 360) - 180)

            if iou_val >= iou_thresh and size_rel_diff <= size_rel_thresh and angle_diff <= angle_thresh_deg:
                considered_same = True
                break

        if considered_same:
            removed_indices.append(i)
        else:
            kept_indices.append(i)

    for i, el in enumerate(last_elems):
        if i in removed_indices:
            continue
        new_A.append(copy.deepcopy(el))

    for i, el_last in enumerate(prev_elems):
        considered_same = False
        bx_l = _get_bbox(el_last)
        kind_l = el_last.get('kind', None)
        size_l = float(el_last.get('size', 0) or 0)
        angle_l = float(el_last.get('angle', 0) or 0)

        for j, el_prev in enumerate(last_elems):
            kind_p = el_prev.get('kind', None)
            if kind_p != kind_l:
                continue
            bx_p = _get_bbox(el_prev)
            iou_val = _iou(bx_l, bx_p)
            size_p = float(el_prev.get('size', 0) or 0)
            size_rel_diff = 0.0
            if max(size_p, size_l) > 0:
                size_rel_diff = abs(size_p - size_l) / max(size_p, size_l)
            angle_p = float(el_prev.get('angle', 0) or 0)
            angle_diff = abs(((angle_l - angle_p + 180) % 360) - 180)

            if iou_val >= iou_thresh and size_rel_diff <= size_rel_thresh and angle_diff <= angle_thresh_deg:
                considered_same = True
                break

        if considered_same:
            removed_indices.append(i)
        else:
            kept_indices.append(i)

    for i, el in enumerate(prev_elems):
        if i in removed_indices:
            continue
        new_A.append(copy.deepcopy(el))

    rule_info.setdefault('op', 'diff_keep')
    rule_info['removed_idx_in_last'] = removed_indices
    rule_info['kept_idx_in_last'] = kept_indices
    rule_info['num_removed'] = len(removed_indices)
    rule_info['num_kept'] = len(kept_indices)
    rule_info.setdefault('seed', seed)

    return new_A, rule_info


def rule_diff_keep_intersection(history_elements, rule_info=None, config=None,
                                 iou_thresh: float = 0.5,
                                 size_rel_thresh: float = 0.2,
                                 angle_thresh_deg: float = 5.0,
                                 use_grid: bool = False,
                                 grid_size: int = 3,
                                 **kwargs):
    """
    去异存同（diff_keep_intersection）— 已修改：
      - 若 len(history_elements) % 3 != 2:
          替换最后一帧 elements 中若干个元素（不能全部替换）。
          如果最后一帧元素数 <=1，则尝试生成并加入一个元素（fallback）。
      - 若 len(history_elements) % 3 == 2:
          只保留最后两帧中被判定为相同的元素（交集）。
    """
    if config is not None and hasattr(config, 'canvas_size'):
        W, H = config.canvas_size
    else:
        W, H = (200, 200)

    A_elems = copy.deepcopy(history_elements[-1]) if len(history_elements) >= 1 else []
    rule_info = rule_info if rule_info else {}

    # preserve seed
    seed = kwargs.get('seed', rule_info.get('seed', None))
    if seed is not None:
        random.seed(seed)
        rule_info['seed'] = seed

    # tolerant populate_prototype
    def _try_populate_prototype():
        bg_color = kwargs.get('bg_color', (255, 255, 255))
        try:
            return populate_prototype(W, H, bg_color=bg_color, seed=seed, use_grid=use_grid, grid_size=grid_size)
        except TypeError:
            try:
                return populate_prototype(W, H, seed=seed, use_grid=use_grid, grid_size=grid_size)
            except Exception:
                return None
        except Exception:
            return None

    def _elems_from_proto(proto):
        if proto is None:
            return []
        if isinstance(proto, dict):
            if 'elements' in proto and isinstance(proto['elements'], (list, tuple)):
                return [copy.deepcopy(e) for e in proto['elements']]
            else:
                return [copy.deepcopy(proto)]
        if isinstance(proto, (list, tuple)):
            return [copy.deepcopy(e) for e in proto]
        return []

    def _create_fallback_elem():
        kind = random.choice(['square', 'circle', 'triangle', 'diamond', 'star'])
        size = random.randint(max(8, min(W, H)//12), max(12, min(W, H)//4))
        cx = random.randint(size//2, max(size//2, W - size//2))
        cy = random.randint(size//2, max(size//2, H - size//2))
        return {
            'kind': kind,
            'size': size,
            'fill': True,
            'center': (cx, cy),
            'angle': random.choice([0,0,45,90]),
            'bbox': (cx-size//2, cy-size//2, size, size),
            'color': kwargs.get('color', None)
        }

    def _get_bbox(el):
        bb = el.get('bbox', None)
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cx, cy = el.get('center', (0,0))
        s = el.get('size', max(1, int(min(W,H)//10)))
        bw = int(max(1, s))
        bh = int(max(1, s))
        bx = int(round(cx - bw/2.0))
        by = int(round(cy - bh/2.0))
        return bx, by, bw, bh

    def _iou(b1, b2):
        bx1, by1, bw1, bh1 = b1
        bx2, by2, bw2, bh2 = b2
        x1 = max(bx1, bx2)
        y1 = max(by1, by2)
        x2 = min(bx1+bw1, bx2+bw2)
        y2 = min(by1+bh1, by2+bh2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2-x1)*(y2-y1)
        area1 = bw1*bh1
        area2 = bw2*bh2
        union = area1+area2-inter
        if union<=0: return 0.0
        return float(inter)/float(union)

    frame_count = len(history_elements)

    # ---------- 新逻辑：当 frame_count % 3 != 2 时，替换最后一帧若干元素 ----------
    if frame_count % 3 != 2:
        rule_info['op'] = 'replace_some_in_last_frame'
        n = len(A_elems)
        if n <= 1:
            proto = _try_populate_prototype()
            new_elem = None
            if proto:
                elems_from_proto = _elems_from_proto(proto)
                if elems_from_proto:
                    new_elem = copy.deepcopy(elems_from_proto[0])
            if new_elem is None:
                new_elem = _create_fallback_elem()
                rule_info['fallback_created'] = True
            A_elems.append(new_elem)
            rule_info['added_idx'] = len(A_elems)-1
            rule_info.setdefault('seed', seed)
            return A_elems, rule_info

        r = random.randint(1, max(1, n-1))
        replaced_indices = random.sample(range(n), r)
        replaced_indices.sort()
        rule_info['num_replaced'] = r
        rule_info['replaced_idx'] = replaced_indices

        new_elems = []
        proto = _try_populate_prototype()
        if proto:
            elems_from_proto = _elems_from_proto(proto)
            for e in elems_from_proto:
                if len(new_elems) >= r: break
                new_elems.append(copy.deepcopy(e))
        while len(new_elems) < r:
            new_elems.append(_create_fallback_elem())
            rule_info.setdefault('fallback_created_count',0)
            rule_info['fallback_created_count'] += 1

        for i, idx in enumerate(replaced_indices):
            A_elems[idx] = new_elems[i]

        rule_info.setdefault('seed', seed)
        return A_elems, rule_info

    # ---------- 原 diff_keep_intersection 逻辑 ----------
    if len(history_elements) < 2:
        rule_info['op'] = 'diff_keep_intersection_skipped_not_enough_history'
        return A_elems, rule_info

    prev_elems = history_elements[-2]
    last_elems = history_elements[-1]

    kept_indices = []
    removed_indices = []

    for i, el_last in enumerate(last_elems):
        is_same = False
        bx_l = _get_bbox(el_last)
        kind_l = el_last.get('kind', None)
        size_l = float(el_last.get('size',0) or 0)
        angle_l = float(el_last.get('angle',0) or 0)

        for el_prev in prev_elems:
            if el_prev.get('kind', None) != kind_l:
                continue
            bx_p = _get_bbox(el_prev)
            iou_val = _iou(bx_l, bx_p)
            size_p = float(el_prev.get('size',0) or 0)
            size_rel_diff = abs(size_p-size_l)/max(size_p,size_l) if max(size_p,size_l)>0 else 0.0
            angle_p = float(el_prev.get('angle',0) or 0)
            angle_diff = abs(((angle_l-angle_p+180)%360)-180)

            if iou_val>=iou_thresh and size_rel_diff<=size_rel_thresh and angle_diff<=angle_thresh_deg:
                is_same = True
                break
        if is_same:
            kept_indices.append(i)
        else:
            removed_indices.append(i)

    new_A = [copy.deepcopy(last_elems[i]) for i in kept_indices]

    if len(new_A)==0 and len(last_elems)>0:
        best_idx = max(range(len(last_elems)), key=lambda i: _get_bbox(last_elems[i])[2]*_get_bbox(last_elems[i])[3])
        new_A.append(copy.deepcopy(last_elems[best_idx]))
        if best_idx in removed_indices: removed_indices.remove(best_idx)
        if best_idx not in kept_indices: kept_indices.append(best_idx)
        rule_info['forced_keep_due_to_empty'] = True

    rule_info.setdefault('op','diff_keep_intersection')
    rule_info['kept_idx_in_last'] = kept_indices
    rule_info['removed_idx_in_last'] = removed_indices
    rule_info['num_kept'] = len(kept_indices)
    rule_info['num_removed'] = len(removed_indices)
    rule_info.setdefault('seed', seed)

    return new_A, rule_info

# ------------------ RULE_MAP ------------------
RULE_MAP = {
    '平移': rule_translate,
    '旋转': rule_rotate,
    '翻转(镜像)': rule_flip,
    '组合': rule_transform_many,

    '直接叠加': rule_direct_overlay,
    '去同存异': rule_diff_keep_same,
    '去异存同': rule_diff_keep_intersection,
    # '规律叠加': rule_pattern_overlay,

    # '封闭开放': rule_attribute_closed_open,
    # '曲直': rule_attribute_curve_straight,

    '单一遍历': rule_traverse_sequence,
    '多遍历': rule_translate,
    '位置遍历': rule_traverse_positions,
    '元素传递': rule_element_transfer,

    # '部分': rule_face_partial,
    # '图群': rule_face_group,
    # '角': rule_face_angle,

    # '内外线': rule_line_inner_outer,
    # '直曲线': rule_line_straight_curved,
    # '平行线': rule_parallel_lines,
    # '笔画': rule_line_strokes,

    # '内外点': rule_point_inner_outer,
    # '交点': rule_intersection_points,
}

def rule_fallback(A, B, **kwargs):
    try:
        img, elems, size = get_canvas_and_elements(A)
        return {'img': img, 'elements': elems, 'canvas_size': size}
    except Exception:
        return {'img': make_blank(128,128,(255,255,255)), 'elements': [], 'canvas_size': (128,128)}

