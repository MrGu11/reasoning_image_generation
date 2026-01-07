"""
verifier.py
增强版基础验证器：
- 原有的 scene/answer 基本校验
- 如果存在 choices & answer_index，则检查 index 有效且 choices[index] 与 answer 匹配（浅比较）
- 返回更丰富的统计信息
"""
import json
import os

def _shallow_equal(a, b):
    # simple dict/list equality for our small objects
    return a == b

def verify_sample(json_path, verbose=False):
    with open(json_path, "r") as f:
        sample = json.load(f)
    template = sample.get("template")
    scene = sample.get("scene", [])
    answer = sample.get("answer")
    ok = True
    reasons = []
    # basic checks
    if not scene or len(scene) == 0:
        ok = False
        reasons.append("empty_scene")
    if answer is None:
        ok = False
        reasons.append("no_answer")
    # choices check
    choices = sample.get("choices")
    if choices is not None:
        ai = sample.get("answer_index")
        if ai is None:
            ok = False
            reasons.append("choices_but_no_answer_index")
        else:
            # index range
            if not (0 <= ai < len(choices)):
                ok = False
                reasons.append("answer_index_out_of_range")
            else:
                # check that the designated choice equals the answer (shallow)
                if not _shallow_equal(choices[ai], answer):
                    ok = False
                    reasons.append("answer_mismatch_with_choice")
    # template-specific quick rules
    if template == "count_increment":
        # expect the last answer to be a list of objects equal to matrix_size
        if not isinstance(answer, list):
            ok = False
            reasons.append("answer_not_list_for_count_increment")
    if verbose:
        print(sample.get("id", os.path.basename(json_path)), "OK" if ok else "FAIL", reasons)
    return ok, reasons

def verify_batch(dir_path):
    results = {"total":0, "ok":0, "fail":0, "fail_reasons":{}}
    for fname in os.listdir(dir_path):
        if not fname.endswith(".json") or fname=="manifest.json":
            continue
        path = os.path.join(dir_path, fname)
        results["total"] += 1
        ok, reasons = verify_sample(path)
        if ok:
            results["ok"] += 1
        else:
            results["fail"] += 1
            for r in reasons:
                results["fail_reasons"][r] = results["fail_reasons"].get(r,0)+1
    return results
