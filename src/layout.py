# layout.py
"""
Image layout / composition helpers for RPMGenerator.

Exports:
- fit_into_cell(src, cell_size) -> square np.ndarray
- make_query_image(cell_size) -> square np.ndarray with '?'
- compose_grid(W, H, states, candidates, sample_dir, g, num_options,
               margin=20, padding_v=20, show_labels=True, show_border=True)
    -> (grid_im, cells_meta, seq_meta, opts_meta, query_path, grid_h, cell_size)
"""
from typing import List, Dict, Tuple, Optional
import os
import numpy as np
import cv2

def fit_into_cell(src: np.ndarray, cell_size: int) -> np.ndarray:
    """Resize src to fit into square cell preserving aspect ratio, centered on white background."""
    Hs, Ws = src.shape[:2]
    if Ws == 0 or Hs == 0 or cell_size <= 0:
        return np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    scale = min(cell_size / Ws, cell_size / Hs)
    new_w = max(1, int(round(Ws * scale)))
    new_h = max(1, int(round(Hs * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(src, (new_w, new_h), interpolation=interp)
    patch = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    ox = (cell_size - new_w) // 2
    oy = (cell_size - new_h) // 2
    patch[oy:oy+new_h, ox:ox+new_w] = resized
    return patch

def make_query_image(cell_size: int, color: Tuple[int,int,int]=(0,0,0)) -> np.ndarray:
    """Create a square image with a big question mark centered."""
    im = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, cell_size // 80)
    font_scale = cell_size / 100.0
    text = '?'
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    if tw > cell_size * 0.8:
        font_scale = font_scale * (cell_size * 0.8 / tw)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (cell_size - tw) // 2
    y = (cell_size + th) // 2
    cv2.putText(im, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return im

def compose_grid(
    W: int,
    H: int,
    states: List[Dict],
    candidates: List[Dict],
    sample_dir: str,
    num_options: int,
    margin: int = 20,
    padding_v: int = 20,
    show_labels: bool = True,
    show_border: bool = True,
    bg_color: Tuple[int,int,int]=(255,255,255)
) -> Tuple[np.ndarray, List[Dict], List[Dict], List[Dict], Optional[str], int, int]:
    """
    Compose the final grid image from states (list of dict with 'state_img','state_path','proto_path')
    and candidates (list of dict with 'img','path','is_correct').

    Returns:
      grid_im, cells_meta, seq_meta, opts_meta, query_path, grid_h, cell_size
    """
    # compute columns for each row separately (sequence includes an extra query cell)
    cols_seq = len(states) + 1  # +1 for query cell
    cols_opt = num_options

    # determine cell_size so both rows fit into H given margins and padding
    max_cell_w = (W - 2*margin) // max(1, max(cols_seq, cols_opt))
    max_cell_h = (H - 2*margin - padding_v) // 2
    cell_size = max(1, min(max_cell_w, max_cell_h))

    seq_row_w = cols_seq * cell_size
    opt_row_w = cols_opt * cell_size

    grid_h = 2*cell_size + padding_v + 2*margin
    # create background canvas (BGR)
    bgr = (bg_color[2], bg_color[1], bg_color[0])
    grid_im = np.full((grid_h, W, 3), bgr, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.25, min(0.8, cell_size / 240.0))
    font_thick = 1
    text_color = (0,0,0)

    cells_meta: List[Dict] = []

    # render sequence row, center horizontally
    seq_offset_x = (W - seq_row_w) // 2
    top_y = margin
    query_saved_path = None

    for i in range(cols_seq):
        dst_x = seq_offset_x + i * cell_size
        dst_y = top_y

        if i < len(states):
            patch = fit_into_cell(states[i]['state_img'], cell_size)
            label = f'S{i}' if show_labels else ''
            proto_path = states[i].get('proto_path')
            state_path = states[i].get('state_path')
            is_query = False
            query_path = None
        elif i == len(states):  # the query cell
            patch = make_query_image(cell_size)
            label = f'S{i}' if show_labels else ''
            query_saved_path = os.path.join(sample_dir, 'query.png')
            # save query image
            cv2.imwrite(query_saved_path, patch)
            proto_path = None
            state_path = None
            is_query = True
            query_path = query_saved_path
        else:
            patch = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
            label = ''
            proto_path = None
            state_path = None
            is_query = False
            query_path = None

        # place
        grid_im[dst_y:dst_y+cell_size, dst_x:dst_x+cell_size] = patch

        if show_border:
            cv2.rectangle(grid_im, (dst_x, dst_y), (dst_x+cell_size-1, dst_y+cell_size-1), (0,0,0), 1)

        if show_labels and label:
            text_x = dst_x + 3
            text_y = dst_y + cell_size + int(12 * font_scale) + 6
            cv2.putText(grid_im, label, (text_x, text_y), font, font_scale, text_color, font_thick, cv2.LINE_AA)

        bbox = [int(dst_x), int(dst_y), int(cell_size), int(cell_size)]
        cells_meta.append({
            'r': 0, 'c': i, 'label': label, 'bbox': bbox,
            'proto_path': proto_path, 'state_path': state_path,
            'is_query': bool(is_query), 'query_path': query_path
        })

    # render options row, center horizontally
    opt_offset_x = (W - opt_row_w) // 2
    bottom_y = top_y + cell_size + padding_v

    # ensure candidates list has at least num_options entries; if not, pad with blanks
    # but we won't create more saved files here; generator is responsible for candidates length
    for i in range(cols_opt):
        dst_x = opt_offset_x + i * cell_size
        dst_y = bottom_y
        if i < num_options:
            candidate = candidates[i]
            patch = fit_into_cell(candidate['img'], cell_size)
            label = chr(65 + i) if show_labels else ''
            opt_path = candidate.get('path')
            is_correct = bool(candidate.get('is_correct', False))
        else:
            patch = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
            label = ''
            opt_path = None
            is_correct = False

        grid_im[dst_y:dst_y+cell_size, dst_x:dst_x+cell_size] = patch

        if show_border:
            cv2.rectangle(grid_im, (dst_x, dst_y), (dst_x+cell_size-1, dst_y+cell_size-1), (0,0,0), 1)

        if show_labels and label:
            text_x = dst_x + 3
            text_y = dst_y + cell_size + int(12 * font_scale) + 6
            cv2.putText(grid_im, label, (text_x, text_y), font, font_scale, text_color, font_thick, cv2.LINE_AA)

        bbox = [int(dst_x), int(dst_y), int(cell_size), int(cell_size)]
        cells_meta.append({
            'r': 1, 'c': i, 'label': label, 'bbox': bbox,
            'path': opt_path, 'is_correct': is_correct
        })

    # build seq_meta and opts_meta for return (sequence meta: include query)
    seq_meta = []
    for s in states:
        seq_meta.append({'proto_path': s.get('proto_path'), 'state_path': s.get('state_path'), 'is_query': False})
    seq_meta.append({'proto_path': None, 'state_path': None, 'is_query': True, 'query_path': query_saved_path})

    opts_meta = []
    for idx, c in enumerate(candidates):
        opts_meta.append({'path': c.get('path'), 'is_correct': bool(c.get('is_correct', False)), 'label': chr(65 + idx)})

    return grid_im, cells_meta, seq_meta, opts_meta, query_saved_path, grid_h, cell_size
