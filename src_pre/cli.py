# cli.py (modified to integrate export_questions.py)
"""
CLI for generation -> rendering -> verification pipeline,
and optional export / overlay of question text using export_questions.py.
"""

import argparse
import os
import json
from tqdm import tqdm

from generator import generate_batch
from renderer import render_sample_json
from verifier import verify_batch

# new: import export utilities
from export_questions import export_questions_to_txt, overlay_question_on_image

def render_all_jsons(json_dir, img_dir, canvas_size=(1200,900), matrix_size=3,
                     highlight_answer=False):
    os.makedirs(img_dir, exist_ok=True)
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json") and f!="manifest.json"])
    for jf in tqdm(json_files, desc="Rendering JSON -> SVG/PNG"):
        jf_path = os.path.join(json_dir, jf)
        # try:
        render_sample_json(jf_path, img_dir, canvas_size=canvas_size,
                               matrix_size=matrix_size, highlight_answer=highlight_answer)
        # except Exception as e:
        #     print(f"Render failed for {jf_path}: {e}")

def overlay_all(json_dir, img_dir, out_dir, font_path=None, font_size=20):
    """
    Pair images and jsons by id (id.png or id.svg expected in img_dir),
    overlay text onto image and save into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json") and f!="manifest.json"])
    for jf in tqdm(json_files, desc="Overlaying questions on images"):
        sid = os.path.splitext(jf)[0]
        # prefer PNG then SVG->PNG (renderer writes PNG if cairosvg installed)
        img_base_png = os.path.join(img_dir, sid + ".png")
        img_base_svg = os.path.join(img_dir, sid + ".svg")
        img_path = None
        if os.path.exists(img_base_png):
            img_path = img_base_png
        elif os.path.exists(img_base_svg):
            # try to rasterize svg to temp png using cairosvg if present (fallback)
            try:
                import cairosvg
                tmp_png = os.path.join(out_dir, sid + "_raster_tmp.png")
                cairosvg.svg2png(url=img_base_svg, write_to=tmp_png)
                img_path = tmp_png
            except Exception:
                print(f"Cannot rasterize SVG for {sid}, skipping overlay (cairosvg missing).")
                continue
        else:
            print(f"Image for {sid} not found in {img_dir}, skipping.")
            continue

        json_path = os.path.join(json_dir, jf)
        out_png = os.path.join(out_dir, sid + "_with_question.png")
        try:
            overlay_question_on_image(img_path, json_path, out_png, font_path=font_path, font_size=font_size)
        except Exception as e:
            print(f"Overlay failed for {sid}: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data0/home/qwen/thgu/Reasoning_generation/data", help="base output dir")
    p.add_argument("--n", type=int, default=100, help="num samples to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only_generate", action="store_true", help="only run generator (no render/verify/export)")
    p.add_argument("--skip_render", action="store_true", help="skip rendering step")
    p.add_argument("--skip_verify", action="store_true", help="skip verification step")
    # export / overlay options
    p.add_argument("--export_questions", action="store_true", help="export questions (text files) from JSONs")
    p.add_argument("--questions_out", type=str, default=None, help="directory to write question txts (defaults to <out>/questions_txt)")
    p.add_argument("--overlay_questions", action="store_true", help="overlay question text on images and save PNGs")
    p.add_argument("--overlay_out", type=str, default=None, help="directory to write overlayed PNGs (defaults to <out>/combined)")
    p.add_argument("--font", type=str, default=None, help="optional TTF font path for overlay")
    p.add_argument("--font_size", type=int, default=20, help="font size for overlay text")
    # rendering options
    p.add_argument("--canvas_w", type=int, default=1200, help="canvas width for rendering")
    p.add_argument("--canvas_h", type=int, default=900, help="canvas height for rendering")
    p.add_argument("--highlight_answer", action="store_true", help="highlight correct choice in rendering")
    args = p.parse_args()

    out = args.out
    json_dir = os.path.join(out, "json")
    images_dir = os.path.join(out, "images")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    if args.only_generate:
        print("Only generating JSON specs...")
        generate_batch(json_dir, n=args.n, seed=args.seed)
        print("Done generation.")
        return

    # 1) generate
    print("Generating JSON specs...")
    generate_batch(json_dir, n=args.n, seed=args.seed)

    # 2) render
    if not args.skip_render:
        print("Rendering images...")
        canvas_size = (args.canvas_w, args.canvas_h)
        render_all_jsons(json_dir, images_dir, canvas_size=canvas_size,
                         matrix_size=None, highlight_answer=args.highlight_answer)
    else:
        print("Skipping rendering as requested.")

    # 3) verify
    if not args.skip_verify:
        print("Verifying JSONs...")
        res = verify_batch(json_dir)
        print("Verify results:", res)
    else:
        print("Skipping verification as requested.")

    # 4) export questions to text
    if args.export_questions:
        q_out = args.questions_out or os.path.join(out, "questions_txt")
        print(f"Exporting questions to text files at {q_out} ...")
        export_questions_to_txt(json_dir, q_out)
    # 5) overlay text onto images
    if args.overlay_questions:
        overlay_out = args.overlay_out or os.path.join(out, "combined")
        print(f"Overlaying question text on images into {overlay_out} ...")
        overlay_all(json_dir, images_dir, overlay_out, font_path=args.font, font_size=args.font_size)

    print("All done.")

if __name__ == "__main__":
    main()
