import json
import random
import os
import sys

def load_json_file(filepath):
    """常规加载.json文件（适用于小文件）。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 -> {filepath}。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 '{filepath}' 不是有效的JSON格式。")
        sys.exit(1)

def count_lines_in_file(filepath):
    """快速计算文件行数，无需加载到内存。"""
    print(f"   -> 正在快速计算 '{os.path.basename(filepath)}' 的总行数...")
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def reservoir_sample_jsonl(filepath, k):
    """
    使用蓄水池抽样从一个大的 .jsonl 文件中高效地随机抽取 k 个样本。
    这只会在内存中保留 k 个元素。
    """
    print(f"   -> 正在从 '{os.path.basename(filepath)}' 中进行蓄水池抽样...")
    reservoir = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < k:
                # 1. 直接填满蓄水池
                reservoir.append(json.loads(line))
            else:
                # 2. 以 k/i 的概率替换旧元素
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = json.loads(line)
    return reservoir

def process_and_mix_datasets(file1_path, file2_path, output_dir, ratios):
    """
    根据给定的比例，高效混合两个JSON文件的数据。
    """
    # --- 1. 识别文件类型并获取大小 ---
    # 我们假设 .jsonl 文件是潜在的大文件，另一个是小文件
    if file1_path.endswith('.jsonl') and file2_path.endswith('.json'):
        large_file_path, small_file_path = file1_path, file2_path
        large_file_is_file1 = True
    elif file2_path.endswith('.jsonl') and file1_path.endswith('.json'):
        large_file_path, small_file_path = file2_path, file1_path
        large_file_is_file1 = False
    else:
        print("❌ 错误: 脚本需要一个 .json 文件和一个 .jsonl 文件才能进行优化。")
        print(f"   文件1: {file1_path}")
        print(f"   文件2: {file2_path}")
        sys.exit(1)

    print(f"识别到小文件 (完全加载): {os.path.basename(small_file_path)}")
    print(f"识别到大文件 (流式处理): {os.path.basename(large_file_path)}")

    # 加载小文件数据，并获取大文件的行数
    small_data = load_json_file(small_file_path)
    large_file_len = count_lines_in_file(large_file_path)

    print(f" -> 小文件加载了 {len(small_data)} 条数据。")
    print(f" -> 大文件共有 {large_file_len} 条数据。")

    # --- 2. 确定基准数量 ---
    # 逻辑保持不变：基准数量由两个文件中较小的那一个决定
    if len(small_data) <= large_file_len:
        base_size = len(small_data)
        base_is_from_small_file = True
        print(f"\n抽样基准由小文件决定，数量 = {base_size}")
    else:
        base_size = large_file_len
        base_is_from_small_file = False
        print(f"\n抽样基准由大文件决定，数量 = {base_size}")

    if base_size == 0:
        print("❌ 错误: 基准文件中没有数据，无法进行混合。")
        sys.exit(1)

    # --- 3. 创建输出目录 (您的代码是正确的) ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到目录: '{output_dir}'")

    # --- 4. 遍历比例进行处理 ---
    for ratio_str in ratios:
        # (解析比例字符串的代码与您的一样，所以这里省略了重复部分)
        r1_str, r2_str = ratio_str.split(':')
        r1, r2 = int(r1_str), int(r2_str)
        print(f"\n--- 开始处理比例 {r1}:{r2} ---")
        
        # 确定r_small, r_large的值
        if base_is_from_small_file:
            # 根据哪个文件是file1/file2来分配比例值
            r_small = r2 if large_file_is_file1 else r1
            r_large = r1 if large_file_is_file1 else r2
        else:
            r_small = r1 if large_file_is_file1 else r2
            r_large = r2 if large_file_is_file1 else r1

        # 计算抽样数量
        if r_small == 0 and r_large > 0:
            n_base = 0; n_other = base_size
        elif r_large == 0 and r_small > 0:
            n_base = base_size; n_other = 0
        elif r_small > 0 and r_large > 0:
            n_base = base_size
            n_other = int(base_size * (r_large / r_small))
        else:
            n_base = 0; n_other = 0

        # 将抽样数映射回 small_data 和 large_file
        n_small, n_large = (n_base, n_other) if base_is_from_small_file else (n_other, n_base)

        # 防止抽样数超过实际数据量
        n_small = min(n_small, len(small_data))
        n_large = min(n_large, large_file_len)

        print(f"计划抽样: {n_small}条 from '{os.path.basename(small_file_path)}', {n_large}条 from '{os.path.basename(large_file_path)}'")

        # 进行抽样
        sample_small = random.sample(small_data, n_small)
        sample_large = reservoir_sample_jsonl(large_file_path, n_large)

        # 合并并打乱
        combined_data = sample_small + sample_large
        random.shuffle(combined_data)
        print(f"混合后总数据量: {len(combined_data)}")

        # 确定哪个是file1的样本
        sample_f1, sample_f2 = (sample_large, sample_small) if large_file_is_file1 else (sample_small, sample_large)
        
        # 保存到文件
        output_filename = f"input_llm_{r1}_{r2}.json"
        output_path = os.path.join(output_dir, output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            print(f"✅ 成功保存到: {output_path}")
        except IOError as e:
            print(f"❌ 错误: 无法写入文件 {output_path}。错误信息: {e}")

    print("\n🎉 所有任务处理完毕！")


if __name__ == "__main__":
    # --- 请在这里配置 ---
    
    # 1. 输入文件路径
    # 脚本会自动识别哪个是.json，哪个是.jsonl
    FILE_1_PATH = "../pipeline/sqlite/results/synthesis_data/input_llm.json"
    FILE_2_PATH = "./input_llm.jsonl"

    # 2. 输出目录
    OUTPUT_DIR = "./results/mixed_datasets"

    # 3. 需要生成的混合比例列表
    RATIOS_TO_GENERATE = [
        "1:0", "0:1", "1:1", "1:2", "2:1", "1:4", "4:1"
    ]
    
    # --- 配置结束，运行脚本 ---
    process_and_mix_datasets(FILE_1_PATH, FILE_2_PATH, OUTPUT_DIR, RATIOS_TO_GENERATE)
