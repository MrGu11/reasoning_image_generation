# generator.py
"""
generator.py

根据 templates 生成题目 JSON，并为常见答案类型生成候选选项（choices）。
若无法自动生成候选，会使用保底策略生成 4 个候选并设置 answer_index。

输出每条 JSON 包含字段：
- id, seed, template, scene, answer, meta
- 若成功生成候选： choices (list) 和 answer_index (int)
"""

import os
import json
import yaml
import copy
import numpy as np
from typing import Tuple, List, Any

from templates import TEMPLATES, list_templates
from utils import ensure_dir, get_rng

DEFAULT_CONFIG = "config.yaml"


def load_config(path: str = None) -> dict:
    path = path or DEFAULT_CONFIG
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_choices_for_answer(answer: Any, rng: np.random.RandomState,
                            matrix_size: int = 3, num_choices: int = 4) -> Tuple[List[Any], int]:
    """
    Try to generate plausible choices for a given 'answer'.
    Returns (choices_list, answer_index). If unable to generate, returns ([], 0).
    Handles two main cases:
      - answer is a dict (single object with 'type' and optional 'angle') -> vary angle/type
      - answer is a list (multiple objects, count-based) -> vary counts
    """
    choices = []

    # Case 1: single object dict (type/angle/size)
    if isinstance(answer, dict):
        base = answer
        # If has angle, produce rotation variants
        if "angle" in base:
            base_angle = int(base.get("angle", 0))
            # candidate angle shifts
            candidate_shifts = [0, 90, 180, 270]
            cand_angles = [ (base_angle + s) % 360 for s in candidate_shifts ]
            # create objects
            for a in cand_angles[:num_choices]:
                obj = copy.deepcopy(base)
                obj["angle"] = int(a)
                choices.append(obj)
        else:
            # try varying type among common primitives
            types = ["circle", "square", "triangle"]
            t0 = base.get("type")
            cand_types = [t0] + [t for t in types if t != t0]
            for t in cand_types[:num_choices]:
                obj = copy.deepcopy(base)
                obj["type"] = t
                choices.append(obj)

        # shuffle choices
        idxs = list(range(len(choices)))
        rng.shuffle(idxs)
        shuffled = [choices[i] for i in idxs]
        # find index of base
        answer_index = None
        for i, c in enumerate(shuffled):
            if c == base:
                answer_index = i
                break
        if answer_index is None:
            answer_index = 0
        return shuffled[:num_choices], int(answer_index)

    # Case 2: list of objects (count-based)
    if isinstance(answer, list):
        if len(answer) == 0:
            return [], 0
        base_count = len(answer)
        counts = []
        # generate counts: base +/- small deltas
        deltas = [0, 1, -1, 2, -2, 3]
        for d in deltas:
            c = base_count + d
            if c <= 0:
                continue
            counts.append(c)
        # make unique and limit
        counts = list(dict.fromkeys(counts))[:num_choices]
        for c in counts:
            proto = answer[0] if isinstance(answer[0], dict) else {"type": "circle", "size": 18, "angle": 0}
            objs = [copy.deepcopy(proto) for _ in range(c)]
            choices.append(objs)
        idxs = list(range(len(choices)))
        rng.shuffle(idxs)
        shuffled = [choices[i] for i in idxs]
        # find which one matches base_count
        answer_index = None
        for i, ch in enumerate(shuffled):
            if isinstance(ch, list) and len(ch) == base_count:
                answer_index = i
                break
        if answer_index is None:
            answer_index = 0
        return shuffled[:num_choices], int(answer_index)

    # unknown type -> cannot generate
    return [], 0


def fallback_generate_choices(answer: Any, rng: np.random.RandomState,
                              matrix_size: int = 3, num_choices: int = 4) -> Tuple[List[Any], int]:
    """
    Fallback (保底) strategy to produce reasonable candidate distractors
    when make_choices_for_answer returns empty.
    """
    choices = []

    # If answer is dict -> vary angles and types
    if isinstance(answer, dict):
        proto = copy.deepcopy(answer)
        base_type = proto.get("type", "square")
        base_angle = int(proto.get("angle", 0)) if "angle" in proto else 0
        # produce variants: same type with different angles, and type swaps
        angle_variants = [(base_angle + s) % 360 for s in [0, 90, 180, 270]]
        for a in angle_variants:
            obj = copy.deepcopy(proto)
            obj["angle"] = int(a)
            choices.append(obj)
        # if not enough, vary types
        if len(choices) < num_choices:
            other_types = [t for t in ["circle", "square", "triangle"] if t != base_type]
            for t in other_types:
                obj = copy.deepcopy(proto)
                obj["type"] = t
                choices.append(obj)
        choices = choices[:num_choices]
        # find answer_index
        answer_index = 0
        for i, c in enumerate(choices):
            if c == proto:
                answer_index = i
                break
        return choices, int(answer_index)

    # If answer is list -> vary counts around base_count
    if isinstance(answer, list) and len(answer) > 0:
        proto = copy.deepcopy(answer[0]) if isinstance(answer[0], dict) else {"type": "circle", "size": 18, "angle": 0}
        base_count = len(answer)
        counts = [max(1, base_count - 1), base_count, base_count + 1, base_count + 2]
        counts = list(dict.fromkeys(counts))[:num_choices]
        for c in counts:
            objs = [copy.deepcopy(proto) for _ in range(c)]
            choices.append(objs)
        answer_index = 0
        for i, ch in enumerate(choices):
            if isinstance(ch, list) and len(ch) == base_count:
                answer_index = i
                break
        return choices, int(answer_index)

    # fallback generic: produce simple single-object choices
    types = ["circle", "square", "triangle", "square"]
    for i in range(num_choices):
        choices.append({"type": types[i % len(types)], "size": 24, "angle": (i * 90) % 360, "color": "black"})
    # default answer index 0
    return choices, 0


def generate_batch(out_dir: str, n: int = 100, templates: List[str] = None,
                   seed: int = 0, config_path: str = None):
    """
    Generate n samples into out_dir. Each sample saved as <id>.json.
    """
    cfg = load_config(config_path)
    ensure_dir(out_dir)
    rng = get_rng(seed)
    if templates is None:
        templates = cfg.get("templates", list_templates())
    catalog = []

    for i in range(n):
        # choose template
        t_name = templates[int(rng.randint(0, len(templates)))]
        gen_fn = TEMPLATES[t_name]
        matrix_size = int(cfg.get("matrix_size", 3))
        # generate scene and answer using template
        matrix, answer, meta = gen_fn(rng, matrix_size=matrix_size)
        sample_id = f"{t_name}_{i:06d}"

        sample = {
            "id": sample_id,
            "seed": int(rng.randint(0, 2**31 - 1)),
            "template": t_name,
            "scene": matrix,
            "answer": answer,
            "meta": meta
        }

        # try to auto-generate choices
        choices, answer_index = make_choices_for_answer(answer, rng, matrix_size=matrix_size, num_choices=4)

        # if automatic generation failed (empty), use fallback strategy
        if not choices:
            choices, answer_index = fallback_generate_choices(answer, rng, matrix_size=matrix_size, num_choices=4)

        # attach choices & answer_index only if choices exist
        if choices:
            sample["choices"] = choices
            sample["answer_index"] = int(answer_index)

        # save sample JSON
        fname = os.path.join(out_dir, sample["id"] + ".json")
        with open(fname, "w") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)

        catalog.append(sample["id"])

    # write manifest
    manifest = {"count": len(catalog), "ids": catalog}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(catalog)} samples to {out_dir}")
