# full_workflow_sharded_sequential.py

import torch
import torch.nn.functional as F
import json
import os
from itertools import combinations
from transformers import AutoModelForCausalLM

# =================================================================================
# 0. 配置区域
# =================================================================================
CONFIG = {
    "model_paths": {
        "initial": "/path/to/models/qwen2.5-14b", # your base model path
        "math": "/path/to/models/fft/qwen2.5-14b/only_math",
        "medical": "/path/to/models/fft/qwen2.5-14b/only_medical",
        "code": "/path/to/models/fft/qwen2.5-14b/only_code"
    },
    # 我们不再手动指定device_map，将使用 "auto" 让transformers自动切分模型
    "delta_storage_file": "all_domain_deltas.pt",
    "output_dir": "final_domain_analysis_results",
    "torch_dtype": torch.bfloat16
}


def stage_one_calculate_and_save_deltas_sharded():
    """
    第一阶段：使用`device_map="auto"`加载跨越多GPU的超大模型，并串行计算增量。
    """
    print("="*80)
    print("### STAGE 1: (自动切分+串行处理) 计算并保存增量向量 ###")
    print("="*80)

    # 1. 加载常驻的、跨越多GPU的初始模型
    print("\n--- 步骤 1.1: 加载并切分初始模型 ---")
    try:
        model_initial = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_paths"]["initial"],
            torch_dtype=CONFIG["torch_dtype"],
            device_map="auto" # 自动将模型切分到所有可用的GPU上
        )
        print("  - 初始模型加载并自动切分成功。")
    except Exception as e:
        print(f"\n错误：加载初始模型失败。错误信息: {e}")
        return False
        
    domain_deltas_in_memory = {}

    # 2. 逐个领域进行处理
    domains_to_process = ["math", "medical", "code"]
    for domain in domains_to_process:
        print("\n" + "-"*50)
        print(f"--- 步骤 1.2: 开始处理 {domain} 领域 ---")
        
        # a. 加载当前领域的模型（同样自动切分）
        print(f"  - 正在加载并切分 {domain} 模型...")
        try:
            model_domain = AutoModelForCausalLM.from_pretrained(
                CONFIG["model_paths"][domain],
                torch_dtype=CONFIG["torch_dtype"],
                device_map="auto"
            )
            print(f"  - {domain} 模型加载并自动切分成功。")
        except Exception as e:
            print(f"错误: 加载 {domain} 模型失败，跳过此领域。错误信息: {e}")
            continue

        # b. 计算增量
        print(f"  - 正在计算 {domain} 领域的增量...")
        with torch.no_grad():
            delta_vectors = {}
            # 获取初始模型所有参数的字典，方便查找
            initial_params_dict = dict(model_initial.named_parameters())
            
            for name, final_param in model_domain.named_parameters():
                if name not in initial_params_dict:
                    continue
                
                initial_param = initial_params_dict[name]
                
                # 【关键】确保两个张量在同一设备上进行计算
                # final_param 已经由`device_map="auto"`放在了某个GPU上
                # 我们需要将对应的 initial_param 移动到同一设备
                initial_param_on_correct_device = initial_param.to(final_param.device)
                
                delta_vec = final_param - initial_param_on_correct_device
                
                # 将计算结果移回CPU内存进行存储，以释放GPU显存
                delta_vectors[name] = delta_vec.cpu()
            
            domain_deltas_in_memory[domain] = delta_vectors
        print(f"  - {domain} 领域增量计算完成。")
        
        # c. 【关键】卸载当前领域模型，释放显存
        print(f"  - 正在从GPU卸载 {domain} 模型...")
        del model_domain
        torch.cuda.empty_cache()
        print(f"  - {domain} 模型已卸载。")
    
    # 3. 卸载初始模型
    print("\n" + "-"*50)
    print(f"--- 步骤 1.3: 卸载初始模型 ---")
    del model_initial
    torch.cuda.empty_cache()
    print("  - 初始模型已卸载。")

    # 4. 保存所有结果到统一文件
    print("\n--- 步骤 1.4: 保存所有增量向量到统一文件 ---")
    torch.save(domain_deltas_in_memory, CONFIG["delta_storage_file"])
    print(f"  - 成功！所有领域的增量向量已保存到: {CONFIG['delta_storage_file']}")
    
    print("\n### STAGE 1 完成 ###")
    return True


def stage_two_analyze_from_file():
    """
    第二阶段：从统一文件中加载增量向量，进行分析并筛选特色参数（此部分代码无需改动）。
    """
    print("\n" + "="*80)
    print("### STAGE 2: 从文件加载并分析领域特异性 ###")
    print("="*80)
    
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # ... (此部分分析代码与之前版本完全相同，此处为简洁省略，实际使用时请保留) ...
    print(f"\n--- 步骤 2.1: 从 {CONFIG['delta_storage_file']} 加载数据 ---")
    try:
        domain_deltas = torch.load(CONFIG['delta_storage_file'])
        print("  - 数据加载成功！")
    except FileNotFoundError:
        print(f"错误: 未找到增量文件 {CONFIG['delta_storage_file']}。请先确保第一阶段已成功运行。")
        return

    print("\n--- 步骤 2.2: 分析领域特异性 ---")
    domains = list(domain_deltas.keys())
    param_names = list(domain_deltas[domains[0]].keys())
    analysis_results = []
    
    print("  - 正在计算两两相似度和特异性得分...")
    for name in param_names:
        param_data = {"parameter": name}
        for domain1, domain2 in combinations(domains, 2):
            vec1 = domain_deltas[domain1][name].flatten()
            vec2 = domain_deltas[domain2][name].flatten()
            similarity = F.cosine_similarity(vec1, vec2, dim=0, eps=1e-8).item()
            sorted_pair = tuple(sorted((domain1, domain2)))
            param_data[f"sim_{sorted_pair[0]}_vs_{sorted_pair[1]}"] = similarity
            
        for domain_focus in domains:
            other_domains = [d for d in domains if d != domain_focus]
            total_sim = sum(
                param_data[f"sim_{tuple(sorted((domain_focus, other)))[0]}_vs_{tuple(sorted((domain_focus, other)))[1]}"]
                for other in other_domains
            )
            param_data[f"{domain_focus}_specificity_score"] = total_sim / len(other_domains)
            
        analysis_results.append(param_data)
        
    print("\n--- 步骤 2.3: 排序并保存结果 ---")
    for domain_focus in domains:
        sorted_list = sorted(analysis_results, key=lambda x: x[f"{domain_focus}_specificity_score"])
        for i, item in enumerate(sorted_list):
            item[f"{domain_focus}_rank"] = i + 1
        output_path = os.path.join(output_dir, f"{domain_focus}_characteristic_params.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_list, f, indent=4)
        print(f"  - 已保存 {domain_focus} 领域的特色参数列表到: {output_path}")
        
    print("\n### STAGE 2 完成 ###")


if __name__ == "__main__":
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: 未找到CUDA设备。此脚本需要GPU环境。")
    else:
        # 执行第一阶段
        if stage_one_calculate_and_save_deltas_sharded():
            # 如果第一阶段成功，执行第二阶段
            stage_two_analyze_from_file()
    
    print("\n所有任务已结束。")