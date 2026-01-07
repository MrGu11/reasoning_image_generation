# export_questions.py
"""
Export textual questions from generated JSON, and optionally overlay question text on image.

Usage examples:
1) Export all JSON -> txt:
   python export_questions.py --json_dir output/json --out_dir output/questions_txt

2) Overlay question text on a single image:
   python export_questions.py --overlay --image output/images/simple_rotate_000000.png \
       --json output/json/simple_rotate_000000.json --out_png output/combined/simple_rotate_000000.png
"""

import os
import json
import textwrap
import argparse
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_SIZE = 20
LINE_SPACING = 6


def make_question_text(sample: dict, wrap_width: int = 80) -> str:
    """
    Construct a human-readable question text from a sample JSON dict.
    This is generic: it reports template/meta, scene summary, choices and answer_index if present.
    You may customize wording per your templates.
    """
    lines = []
    sid = sample.get("id", "unknown")
    lines.append(f"ID: {sid}")
    lines.append(f"Template: {sample.get('template', '')}")
    meta = sample.get("meta")
    if meta:
        lines.append("Meta: " + json.dumps(meta, ensure_ascii=False))
    lines.append("")  # blank

    # Scene summary: for each cell, show short description
    scene = sample.get("scene", [])
    matrix_size = int(len(scene) ** 0.5) if scene else 0
    lines.append("Scene (cell summaries):")
    for i, cell in enumerate(scene):
        # short description of cell
        if not cell:
            desc = "(blank)"
        else:
            parts = []
            for obj in cell:
                typ = obj.get("type", "obj")
                angle = obj.get("angle")
                size = obj.get("size")
                if angle is not None:
                    parts.append(f"{typ}@{int(angle)}°")
                else:
                    parts.append(f"{typ}")
            desc = "+".join(parts)
        if matrix_size:
            r = i // matrix_size
            c = i % matrix_size
            lines.append(f"  cell[{r},{c}]: {desc}")
        else:
            lines.append(f"  cell[{i}]: {desc}")
    lines.append("")

    # Choices
    choices = sample.get("choices") or sample.get("candidates") or []
    if choices:
        lines.append("Choices:")
        for idx, ch in enumerate(choices):
            # make short string for choice
            if isinstance(ch, dict):
                desc = obj_short_desc(ch)
            elif isinstance(ch, list):
                # if it's a scene, try compact representation
                if len(ch) == matrix_size * matrix_size and matrix_size > 0:
                    desc_items = []
                    for j, cell in enumerate(ch):
                        if not cell:
                            desc_items.append(" . ")
                        else:
                            desc_items.append(obj_short_desc(cell[0]) if isinstance(cell[0], dict) else "obj")
                    desc = "[" + ",".join(desc_items) + "]"
                else:
                    desc = f"{len(ch)} objects"
            else:
                desc = str(ch)
            label = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
            lines.append(f"  {label}. {desc}")
        lines.append("")
    else:
        lines.append("Choices: (none)")
        lines.append("")

    # Answer
    if "answer_index" in sample:
        ai = sample.get("answer_index")
        label = chr(ord("A") + ai) if ai < 26 else str(ai + 1)
        lines.append(f"Answer index: {ai} ({label})")
    else:
        lines.append("Answer index: (not available / open)")

    # wrap long meta lines
    wrapped = []
    for ln in lines:
        if len(ln) > wrap_width:
            wrapped.extend(textwrap.wrap(ln, width=wrap_width))
        else:
            wrapped.append(ln)
    return "\n".join(wrapped)


def obj_short_desc(obj: dict) -> str:
    typ = obj.get("type", "obj")
    angle = obj.get("angle")
    size = obj.get("size")
    if angle is not None:
        return f"{typ}@{int(angle)}°"
    elif size is not None:
        return f"{typ}({int(size)})"
    else:
        return typ


def export_questions_to_txt(json_dir: str, out_dir: str, wrap_width: int = 80):
    """
    For each JSON in json_dir, write a corresponding text file in out_dir
    and write a combined TSV mapping: id \t json_path \t txt_path \t has_choices
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        txt = make_question_text(sample, wrap_width=wrap_width)
        sid = sample.get("id", os.path.splitext(fname)[0])
        out_txt = os.path.join(out_dir, sid + ".txt")
        with open(out_txt, "w", encoding="utf-8") as fo:
            fo.write(txt)
        has_choices = int(bool(sample.get("choices") or sample.get("candidates")))
        rows.append((sid, path, out_txt, has_choices))
    # write TSV manifest
    tsv_path = os.path.join(out_dir, "questions_manifest.tsv")
    with open(tsv_path, "w", encoding="utf-8") as tf:
        tf.write("id\tjson_path\ttxt_path\thas_choices\n")
        for r in rows:
            tf.write("\t".join([r[0], r[1], r[2], str(r[3])]) + "\n")
    print(f"Wrote {len(rows)} question txt files to {out_dir}")
    return tsv_path


def overlay_question_on_image(image_path: str, json_path: str, out_path: str,
                              font_path: Optional[str] = None, font_size: int = DEFAULT_FONT_SIZE,
                              max_width: Optional[int] = None):
    """
    Draw the question text under the image and save to out_path (PNG).
    - image_path: existing rendered image (PNG recommended)
    - json_path: corresponding JSON file to extract question text
    - font_path: path to .ttf font (optional). If None, use default PIL font.
    - font_size: base font size
    - max_width: if provided, wrap text to max_width pixels (approx)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        sample = json.load(f)
    text = make_question_text(sample, wrap_width=80)

    # load image
    im = Image.open(image_path).convert("RGBA")
    iw, ih = im.size

    # choose font
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # estimate text block size: do simple wrap by characters
    # Convert text into lines that fit in image width (approx using character count)
    if max_width is None:
        max_width = iw - 40
    # approximate characters per line
    avg_char_w = font.getsize("M")[0] if hasattr(font, "getsize") else font_size * 0.6
    chars_per_line = max(20, int(max_width / max(1, avg_char_w)))
    wrapped_lines = []
    for paragraph in text.split("\n"):
        wrapped_lines.extend(textwrap.wrap(paragraph, width=chars_per_line) or [""])
    # compute text block height
    line_h = font.getsize("Ay")[1] + LINE_SPACING if hasattr(font, "getsize") else font_size + LINE_SPACING
    text_h = line_h * len(wrapped_lines) + 20

    # create new image: stack vertical (image top, text bottom) with padding
    new_h = ih + text_h
    new_im = Image.new("RGBA", (iw, new_h), (255, 255, 255, 255))
    new_im.paste(im, (0, 0))

    draw = ImageDraw.Draw(new_im)
    # draw each line centered
    y = ih + 10
    for ln in wrapped_lines:
        w = draw.textsize(ln, font=font)[0]
        x = max(10, (iw - w) // 2)
        draw.text((x, y), ln, fill="black", font=font)
        y += line_h

    # save
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    new_im.convert("RGB").save(out_path, format="PNG")
    return out_path


# ----------------- CLI -----------------
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", type=str, help="directory of sample json files")
    p.add_argument("--out_dir", type=str, default="output/questions_txt", help="where to write txt files")
    p.add_argument("--overlay", action="store_true", help="overlay text on image (requires --image and --json)")
    p.add_argument("--image", type=str, help="image path to overlay")
    p.add_argument("--json", type=str, help="json path for overlay")
    p.add_argument("--out_png", type=str, help="output png path for overlay")
    p.add_argument("--font", type=str, help="optional ttf font path")
    args = p.parse_args()

    if args.overlay:
        if not args.image or not args.json or not args.out_png:
            print("For overlay mode, provide --image, --json and --out_png")
            return
        overlay_question_on_image(args.image, args.json, args.out_png, font_path=args.font)
        print("Wrote overlay image:", args.out_png)
    else:
        if not args.json_dir:
            print("Provide --json_dir to export questions.")
            return
        tsv = export_questions_to_txt(args.json_dir, args.out_dir)
        print("Wrote questions and manifest:", tsv)


if __name__ == "__main__":
    _cli()
