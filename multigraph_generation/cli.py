import os
import random
import multiprocessing
from generator import GeometryGenerator

def generate_single(args):
    """单个样本的生成函数，供多进程调用"""
    i, global_scale, log_level, mode = args
    
    # 为每个进程创建独立的日志文件（避免多进程日志冲突）
    pid = os.getpid()
    
    # 每个进程创建独立的生成器实例
    generator = GeometryGenerator(
        global_scale=global_scale,
        log_level=log_level,
    )
    
    # 执行生成任务
    record = generator.generate(
        mode=mode,
        save_path=f"output/images/{i}_{mode}.png",
        params_save_path=f"output/params/{i}_{mode}.json",
        dpi=200,
        seed=i  # 固定种子确保可复现
    )
    
    return f"生成完成：ID={record.generation_id}, 形状数量={record.shape_count}, 索引={i}"

def generate_all(use_multiprocessing=True):
    # 配置参数
    global_scale = 1.3
    log_level = "INFO"
    num_samples = 100
    
    # 创建输出目录
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/params", exist_ok=True)
    
    # 预先生成所有mode（确保与单进程模式下的选择顺序一致）
    # modes = [random.choice(["random", "nested", "adjacent", "intersecting"]) for _ in range(num_samples)]
    modes = [random.choice(["adjacent"]) for _ in range(num_samples)]
    
    # 准备任务参数列表
    tasks = [
        (i, global_scale, log_level, modes[i])
        for i in range(num_samples)
    ]
    
    if use_multiprocessing:
        # 启动进程池（使用CPU核心数作为进程数）
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # 并行执行任务并获取结果
            results = pool.map(generate_single, tasks)
            
            # 打印结果（保持顺序）
            for result in results:
                print(result)
    else:
        # 在单进程模式下执行任务
        for i, task in enumerate(tasks):
            if i != 5:
                continue
            result = generate_single(task)
            print(result)
    
    print("所有生成任务完成")

if __name__ == "__main__":
    # 设置是否使用多进程（True为并行，False为单进程）
    use_multiprocessing = False  # 或 False，根据需要选择
    
    generate_all(use_multiprocessing=use_multiprocessing)