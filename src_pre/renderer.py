# renderer.py
"""
Renderer for RPM-like layout with clear cell borders and option row with letter labels.

Usage:
  render_sample_json("output/json/simple_rotate_000000.json", "output/images")

Expect sample JSON to contain:
{
  "id": "xxx",
  "scene": [ cell0, cell1, ... ]   # length = matrix_size * matrix_size
  "choices": [ choice0, choice1, ... ]  # each choice can be a list-of-objects (single-cell) or a scene
  "answer_index": 2   # optional
}

Key features:
- white background (no stray squares)
- outer border and cell borders for matrix
- divider line between question matrix and choice row
- each choice drawn inside its box with letter label centered below
- optional highlight for correct choice (highlight_answer=True)
"""

import os
import json
import math
import svgwrite
from typing import Tuple

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

LETTER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---- low-level primitive drawing ----
def _stroke_kwargs(color="black", width=2):
    return {"stroke": color, "stroke_width": width, "fill": "none",
            "stroke_linejoin": "round", "stroke_linecap": "round"}

def draw_primitive(dwg: svgwrite.Drawing, obj: dict, x: float, y: float):
    typ = obj.get("type", "circle")
    size = float(obj.get("size", 20))
    color = obj.get("color", "black")
    angle = float(obj.get("angle", 0))
    skw = _stroke_kwargs(color=color, width=2)
    if typ == "circle":
        dwg.add(dwg.circle(center=(x, y), r=size/2, **skw))
    elif typ == "square":
        rect = dwg.rect(insert=(x - size/2, y - size/2), size=(size, size), **skw)
        rect.rotate(angle, center=(x, y))
        dwg.add(rect)
    elif typ == "triangle":
        h = size * (3**0.5) / 2
        points = [(x, y - 2*h/3), (x - size/2, y + h/3), (x + size/2, y + h/3)]
        poly = dwg.polygon(points=points, **skw)
        poly.rotate(angle, center=(x, y))
        dwg.add(poly)
    else:
        # fallback small circle
        dwg.add(dwg.circle(center=(x, y), r=size/3, **skw))


# ---- matrix rendering ----
def render_matrix(dwg: svgwrite.Drawing, scene: list, origin: Tuple[float,float],
                  matrix_size: int, cell_w: float, cell_h: float,
                  draw_cell_borders: bool = True, cell_border_width: float = 3):
    ox, oy = origin
    total_cells = matrix_size * matrix_size
    # outer border rectangle (around full matrix)
    outer_x = ox
    outer_y = oy
    outer_w = cell_w * matrix_size
    outer_h = cell_h * matrix_size
    # draw outer border (thicker)
    dwg.add(dwg.rect(insert=(outer_x, outer_y), size=(outer_w, outer_h),
                     fill="white", stroke="black", stroke_width=max(2, cell_border_width+1),
                     rx=0, ry=0))
    # draw cell borders (internal gridlines)
    if draw_cell_borders:
        # vertical lines
        for c in range(1, matrix_size):
            x = ox + c * cell_w
            dwg.add(dwg.line(start=(x, oy), end=(x, oy + outer_h), stroke="black", stroke_width=cell_border_width,
                             stroke_linejoin="round", stroke_linecap="round"))
        # horizontal lines
        for r in range(1, matrix_size):
            y = oy + r * cell_h
            dwg.add(dwg.line(start=(ox, y), end=(ox + outer_w, y), stroke="black", stroke_width=cell_border_width,
                             stroke_linejoin="round", stroke_linecap="round"))
    # draw objects in each cell
    for idx in range(total_cells):
        cell = scene[idx] if idx < len(scene) else []
        r = idx // matrix_size
        c = idx % matrix_size
        cx = ox + c * cell_w + cell_w / 2
        cy = oy + r * cell_h + cell_h / 2
        if not cell:
            continue
        n = len(cell)
        spacing = min(cell_w / max(n + 1, 1), 30)
        for j, obj in enumerate(cell):
            px = cx + (j - (n - 1) / 2) * spacing
            py = cy
            draw_primitive(dwg, obj, px, py)


# ---- choice rendering ----
def render_choice_box(dwg: svgwrite.Drawing, box_x: float, box_y: float, box_w: float, box_h: float,
                      stroke_w: float = 2, highlight: bool = False):
    if highlight:
        dwg.add(dwg.rect(insert=(box_x - 4, box_y - 4), size=(box_w + 8, box_h + 8),
                         fill="none", stroke="green", stroke_width=4, rx=4, ry=4))
    dwg.add(dwg.rect(insert=(box_x, box_y), size=(box_w, box_h),
                     fill="white", stroke="black", stroke_width=stroke_w, rx=3, ry=3))


def render_choice_content(dwg: svgwrite.Drawing, choice, box_x: float, box_y: float,
                          box_w: float, box_h: float, matrix_size: int):
    cx = box_x + box_w / 2
    cy = box_y + box_h / 2
    # if choice is a full scene (matrix), render scaled tiny matrix
    if isinstance(choice, list) and len(choice) == matrix_size * matrix_size:
        tiny_w = box_w / matrix_size
        tiny_h = box_h / matrix_size
        for idx, cell in enumerate(choice):
            r = idx // matrix_size
            c = idx % matrix_size
            tcx = box_x + c * tiny_w + tiny_w / 2
            tcy = box_y + r * tiny_h + tiny_h / 2
            if not cell:
                continue
            n = len(cell)
            spacing_small = min(tiny_w / max(n + 1, 1), 12)
            for j, obj in enumerate(cell):
                ox = tcx + (j - (n - 1) / 2) * spacing_small
                oy = tcy
                scaled = dict(obj)
                if "size" in scaled:
                    scaled["size"] = max(6, scaled["size"] * (tiny_w / 60.0))
                draw_primitive(dwg, scaled, ox, oy)
    elif isinstance(choice, list) and all(isinstance(o, dict) for o in choice):
        # list of objects in center
        n = len(choice)
        spacing_small = min(box_w / max(n + 1, 1), 20)
        for j, obj in enumerate(choice):
            ox = cx + (j - (n - 1) / 2) * spacing_small
            oy = cy
            scaled = dict(obj)
            if "size" in scaled:
                scaled["size"] = max(6, scaled["size"] * (box_w / 120.0))
            draw_primitive(dwg, scaled, ox, oy)
    elif isinstance(choice, dict):
        scaled = dict(choice)
        if "size" in scaled:
            scaled["size"] = max(6, scaled["size"] * (box_w / 120.0))
        draw_primitive(dwg, scaled, cx, cy)
    else:
        # fallback placeholder
        dwg.add(dwg.text("?", insert=(cx, cy), text_anchor="middle", alignment_baseline="central", font_size=20))


# ---- main public function ----
def render_sample_json(json_path: str, out_dir: str,
                       canvas_size: Tuple[int, int] = (800, 600),
                       matrix_size: int = 3,
                       cell_border_width: float = 3,
                       choice_box_stroke: float = 2,
                       choices_per_row: int = 4,
                       choice_box_size: Tuple[int, int] = (140, 140),
                       divider_thickness: float = 4,
                       highlight_answer: bool = False):
    """
    Render sample JSON into SVG (and optional PNG if cairosvg available).
    - canvas_size: total canvas (w,h)
    - top: main matrix area (we allocate ~45% height for matrix)
    - bottom: choices area
    """
    with open(json_path, "r") as f:
        sample = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    sample_id = sample.get("id", os.path.splitext(os.path.basename(json_path))[0])
    svg_path = os.path.join(out_dir, sample_id + ".svg")
    w, h = canvas_size
    dwg = svgwrite.Drawing(svg_path, size=(w, h))

    # white background (avoid artifacts)
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white", stroke="none"))

    # layout
    pad = 24
    top_h = int(h * 0.45)
    top_w = w - 2 * pad
    cell_w = top_w / matrix_size
    cell_h = top_h / matrix_size
    matrix_origin = (pad, pad)

    # render matrix (with outer border + cell lines)
    render_matrix(dwg, sample.get("scene", []), matrix_origin, matrix_size, cell_w, cell_h,
                  draw_cell_borders=True, cell_border_width=cell_border_width)

    # draw divider line under matrix
    divider_y = pad + top_h + 12
    dwg.add(dwg.line(start=(pad, divider_y), end=(w - pad, divider_y),
                     stroke="black", stroke_width=divider_thickness))

    # choices area
    choices = sample.get("choices", []) or sample.get("candidates", [])
    if choices:
        num_choices = len(choices)
        cols = min(choices_per_row, num_choices)
        rows = math.ceil(num_choices / cols)
        # compute horizontal start so choices centered
        box_w, box_h = choice_box_size
        total_w = cols * box_w + (cols - 1) * 20  # 20 px gap
        start_x = (w - total_w) / 2
        # vertical start under divider
        start_y = divider_y + 20
        # render each choice box and content
        for i, choice in enumerate(choices):
            col = i % cols
            row = i // cols
            bx = start_x + col * (box_w + 20)
            by = start_y + row * (box_h + 50)  # 50 px for label text area below
            is_correct = (sample.get("answer_index") is not None) and (i == sample.get("answer_index"))
            # draw choice box (with optional highlight)
            render_choice_box(dwg, bx, by, box_w, box_h, stroke_w=choice_box_stroke, highlight=(highlight_answer and is_correct))
            # draw content
            render_choice_content(dwg, choice, bx, by, box_w, box_h, matrix_size)
            # draw letter label under box (centered)
            label = LETTER_LABELS[i] if i < len(LETTER_LABELS) else str(i + 1)
            lx = bx + box_w / 2
            ly = by + box_h + 30
            dwg.add(dwg.text(label, insert=(lx, ly), text_anchor="middle", font_size=28, fill="black"))

    dwg.save()

    # optional PNG export
    if CAIROSVG_AVAILABLE:
        png_path = os.path.join(out_dir, sample_id + ".png")
        # export at same resolution as canvas_size
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=w, output_height=h)

    return svg_path



