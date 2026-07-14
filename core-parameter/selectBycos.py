# find_characteristic_params_multi_gpu_FIXED.py

import torch
import torch.nn.functional as F
import json
import os
from itertools import combinations
from transformers import AutoModelForCausalLM

# =================================================================================
# 0. 配置区域 (针对多GPU环境)
# =================================================================================
CONFIG = {
    "model_paths": {
        "initial": "/path/to/models/llama3-8b",
        "gsm8k": "/path/to/models/fft/llama3-8b/gsm8k",
        "math": "/path/to/models/fft/llama3-8b/math",
        "code": "/path/to/models/fft/llama3-8b/only_code"
    },
    "device_map": {
        "initial": "cuda:0",
        "gsm8k": "cuda:1",
        "math": "cuda:2",
        "code": "cuda:3"
    },
    "output_dir": "llama3-8b/newMath",
    "torch_dtype": torch.bfloat16
}


def main():
    """
    主执行函数，完成从模型加载到特色参数筛选的全过程 (多GPU并行优化版)。
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        print("错误: 此脚本需要至少4个CUDA设备。请检查您的环境和CONFIG配置。")
        return
        
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # ... (步骤1 和 步骤2 代码与之前完全相同，此处省略以保持简洁) ...
    # 假设之前的模型加载和增量计算代码已成功执行

    # =================================================================================
    # 1. 并行化加载模型到指定GPU (代码同上)
    # =================================================================================
    print("--- 步骤 1: 开始并行加载模型到指定GPU ---")
    
    models = {}
    try:
        for name, path in CONFIG["model_paths"].items():
            device = CONFIG["device_map"][name]
            print(f"  - 正在加载 {name} 模型到 {device}...")
            models[name] = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=CONFIG["torch_dtype"],
                device_map=device
            )
        print(f"--- 所有模型加载完毕 ---\n")

    except Exception as e:
        print(f"\n错误：模型加载失败。错误信息: {e}")
        return

    # =================================================================================
    # 2. 并行计算每个领域的参数增量向量 (代码同上)
    # =================================================================================
    print("--- 步骤 2: 在各GPU上并行计算参数增量向量 ---")
    
    domain_deltas = {}
    initial_params_on_devices = {}
    
    with torch.no_grad():
        for domain, model in models.items():
            if domain == "initial":
                continue
            target_device = CONFIG["device_map"][domain]
            print(f"  - 正在将初始参数分发到 {target_device}...")
            initial_params_on_devices[domain] = {
                name: p.to(target_device) for name, p in models["initial"].named_parameters()
            }

        for domain, model in models.items():
            if domain == "initial":
                continue
            
            target_device = CONFIG["device_map"][domain]
            print(f"  - 正在 {target_device} 上处理 {domain} 领域...")
            
            delta_vectors = {}
            initial_params_local = initial_params_on_devices[domain]
            
            for name, final_param in model.named_parameters():
                if name not in initial_params_local:
                    continue
                delta_vectors[name] = final_param - initial_params_local[name]
            
            domain_deltas[domain] = {name: d.cpu() for name, d in delta_vectors.items()}
            print(f"  - {domain} 领域在 {target_device} 上计算完成。")
            
            del initial_params_on_devices[domain], delta_vectors, initial_params_local
            torch.cuda.empty_cache()

    print("--- 所有增量向量计算完毕 ---\n")
    
    # =================================================================================
    # 3. 领域特异性分析 (此部分为修正后的代码)
    # =================================================================================
    print("--- 步骤 3: 在CPU上分析领域特异性 ---")
    
    domains = list(domain_deltas.keys())
    param_names = list(domain_deltas[domains[0]].keys())
    analysis_results = []
    
    print("  - 正在计算两两相似度和特异性得分...")
    for i, name in enumerate(param_names):
        param_data = {"parameter": name}
        
        # a. 计算所有两两组合的余弦相似度
        for domain1, domain2 in combinations(domains, 2):
            vec1 = domain_deltas[domain1][name].flatten()
            vec2 = domain_deltas[domain2][name].flatten()
            similarity = F.cosine_similarity(vec1, vec2, dim=0, eps=1e-8).item()
            
            # 【修正点】始终使用排序后的元组来创建键，确保键名规范化
            sorted_pair = tuple(sorted((domain1, domain2)))
            param_data[f"sim_{sorted_pair[0]}_vs_{sorted_pair[1]}"] = similarity
            
        # b. 为每个领域计算其“特异性得分”
        for domain_focus in domains:
            other_domains = [d for d in domains if d != domain_focus]
            total_sim = 0
            for other in other_domains:
                # 【修正点】查找键时也使用排序后的元组，与创建时保持一致
                sorted_pair = tuple(sorted((domain_focus, other)))
                sim_key = f"sim_{sorted_pair[0]}_vs_{sorted_pair[1]}"
                total_sim += param_data[sim_key] # 现在可以保证找到键
            
            param_data[f"{domain_focus}_specificity_score"] = total_sim / len(other_domains)
            
        analysis_results.append(param_data)
        
    print("--- 特异性分析完成 ---\n")

    # =================================================================================
    # 4. 排序并为每个领域保存结果 (代码同上)
    # =================================================================================
    print("--- 步骤 4: 排序并保存特色参数列表 ---")
    
    for domain_focus in domains:
        sorted_list = sorted(
            analysis_results,
            key=lambda x: x[f"{domain_focus}_specificity_score"]
        )
        
        for i, item in enumerate(sorted_list):
            item[f"{domain_focus}_rank"] = i + 1

        output_path = os.path.join(CONFIG["output_dir"], f"{domain_focus}_characteristic_params.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_list, f, indent=4)
        print(f"  - 已保存 {domain_focus} 领域的特色参数列表到: {output_path}")
    
    # =================================================================================
    # 5. 提取并保存top r%参数
    # =================================================================================
    print("\n--- 步骤 5: 提取并保存top r%参数 ---")
    
    top_percentages = [0.05, 0.15, 0.20]  # 5%, 15%, 20%
    
    for domain_focus in domains:
        # 读取已保存的特色参数列表
        output_path = os.path.join(CONFIG["output_dir"], f"{domain_focus}_characteristic_params.json")
        with open(output_path, 'r', encoding='utf-8') as f:
            sorted_list = json.load(f)
        
        total_params = len(sorted_list)
        
        for r in top_percentages:
            top_n = int(total_params * r)
            if top_n < 1:
                top_n = 1
            
            # 提取top r%参数（根据rank排序，rank越小越具有领域特异性）
            top_params = [
                {
                    "parameter": item["parameter"],
                    "rank": item[f"{domain_focus}_rank"],
                    "specificity_score": item[f"{domain_focus}_specificity_score"]
                }
                for item in sorted_list[:top_n]
            ]
            
            # 保存top r%参数
            top_output_path = os.path.join(CONFIG["output_dir"], f"{domain_focus}_top_{int(r*100)}percent.json")
            with open(top_output_path, 'w', encoding='utf-8') as f:
                json.dump(top_params, f, indent=4)
            print(f"  - 已保存 {domain_focus} 领域 top {int(r*100)}% 参数到: {top_output_path} (共 {top_n} 个参数)")
        
    print("--- 所有任务完成！ ---")


if __name__ == "__main__":
    main()