#!/usr/bin/env python3
import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

# 导入你的模块（确保这些模块在 PYTHONPATH 中可导入）
from config import GenConfig
from generator import RPMGenerator

def _make_sample(index: int, out_dir: str, grid: int, seed: Optional[int]) -> Dict[str, Any]:
    """
    在子进程中运行：为每个样本创建独立的 GenConfig + RPMGenerator 并生成样本。
    返回 meta dict（或包含 error 字段的 dict）。
    """
    try:
        # 每个样本使用不同的种子以避免重复（如果 seed 为 None，则不设置）
        local_seed = None if seed is None else seed + index
        cfg = GenConfig(out_dir=out_dir, grid_size=grid, seed=local_seed)
        gen = RPMGenerator(cfg)
        meta = gen.generate_sample(index)
        return meta
    except Exception as e:
        # 返回一个包含错误信息的字典，主进程可以记录或处理
        tb = traceback.format_exc()
        return {
            "index": index,
            "error": True,
            "error_type": str(type(e)),
            "error_message": str(e),
            "traceback": tb
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='../data_251027_2w')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--grid', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行 worker 数（默认为 CPU 核心数）。若你的生成器使用 GPU，请设置为1。')
    parser.add_argument('--use_threads', action='store_true',
                        help='使用线程池而不是进程池（适合 I/O 密集或不可 picklable 对象）。')
    return parser.parse_args()

def write_index(out_dir: str, metas):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()

    if args.test:
        # 保持 test 模式为简单顺序运行（便于调试）
        cfg = GenConfig(out_dir='.', grid_size=3, seed=42)
        gen = RPMGenerator(cfg)
        metas = []
        for i in range(3):
            meta = gen.generate_sample(i)
            metas.append(meta)
        # basic assertions
        for m in metas:
            assert os.path.exists(m['saved']), 'sample dir missing'
            assert os.path.exists(os.path.join(m['saved'],'grid.png'))
            assert os.path.exists(os.path.join(m['saved'],'meta.json'))
            assert os.path.exists(os.path.join(m['saved'],'coco.json'))
        print('Integration test passed, samples in current directory')
        return

    out_dir = args.out_dir
    n = args.n
    grid = args.grid
    seed = args.seed

    os.makedirs(out_dir, exist_ok=True)

    # 选择执行器类型
    max_workers = args.workers if args.workers and args.workers > 0 else os.cpu_count() or 1
    use_threads = args.use_threads

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    print(f"Start generating {n} samples -> {out_dir} using "
          f"{'threads' if use_threads else 'processes'} (workers={max_workers}), grid={grid}, seed={seed}")

    metas = [None] * n
    futures = {}

    # 试着使用 tqdm，如果没有则回退（不强制依赖）
    try:
        from tqdm import tqdm
        has_tqdm = True
    except Exception:
        has_tqdm = False

    # Submit tasks
    with executor_cls(max_workers=max_workers) as exe:
        for i in range(n):
            fut = exe.submit(_make_sample, i, out_dir, grid, seed)
            futures[fut] = i

        # iterate completed futures and collect metas
        if has_tqdm:
            pbar = tqdm(total=n)
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                # should be rare because _make_sample 捕获异常并返回错误字典
                res = {
                    "index": idx,
                    "error": True,
                    "error_type": str(type(e)),
                    "error_message": str(e),
                }
            metas[idx] = res
            completed += 1
            if has_tqdm:
                pbar.update(1)
            else:
                if completed % max(1, (n // 10)) == 0 or completed == n:
                    print(f"Progress: {completed}/{n}")
        if has_tqdm:
            pbar.close()

    # 保存 index
    write_index(out_dir, metas)
    print('Done. Generated', len(metas), 'samples to', out_dir)

if __name__ == '__main__':
    main()
