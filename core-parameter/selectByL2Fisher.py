import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ==== 全局设置 ====
device = "cuda" if torch.cuda.is_available() else "cpu"
alpha, beta = 0.5, 0.5  # L2 和 Fisher 的权重
top_k_ratio = 0.1       # 筛选前 10% 的参数

# ==== 1. 数据集与工具函数定义 (保持不变) ====

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/models/llama3-8b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class InstructDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 根据文件扩展名选择读取方式
        if jsonl_path.endswith('.jsonl'):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line.strip()))
        elif jsonl_path.endswith('.json'):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    self.data = json_data
                else:
                    self.data = [json_data]
        else:
            raise ValueError(f"不支持的文件格式: {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 兼容多种数据格式
                # 1) problem / solution
        if "problem" in item and "solution" in item:
            instruction = item.get("problem", "")
            output = item.get("solution", "")
            input_text = instruction

        # 2) instruction / output (+ 可选 input)
        elif "instruction" in item and "output" in item:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            if item.get("input", "").strip():
                input_text = f"{instruction}\n{item['input']}"
            else:
                input_text = instruction
         # 3) question / answer（比如 grade_school_math）
        elif "question" in item and "answer" in item:
            instruction = item.get("question", "")
            output = item.get("answer", "")
            input_text = instruction
        else:
            raise ValueError(
                f"数据格式不支持: {item}，需要包含 (problem, solution) 或 (instruction, output)"
            )
        
        full_text = f"{input_text}\n{output}"
        
        # 编码
        input_encoding = self.tokenizer(
            input_text, 
            truncation=True, 
            max_length=self.max_length-100,
            return_tensors='pt'
        )
        
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_length = input_encoding['input_ids'].shape[1]
        
        return {
            'input_ids': full_encoding['input_ids'].squeeze(0),
            'attention_mask': full_encoding['attention_mask'].squeeze(0),
            'input_length': input_length
        }

def create_dataloader(jsonl_path, batch_size=1):
    dataset = InstructDataset(jsonl_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def sft_loss_fn(logits, input_ids, attention_mask, input_length):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention = attention_mask[..., 1:].contiguous()
    
    loss_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    for i in range(shift_labels.shape[0]):
        valid_len = min(input_length[i], shift_labels.shape[1])
        loss_mask[i, valid_len:] = True
    
    final_mask = shift_attention.bool() & loss_mask
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    losses = losses.view(shift_labels.shape)
    
    masked_losses = losses * final_mask.float()
    
    if final_mask.sum() > 0:
        return masked_losses.sum() / final_mask.sum()
    else:
        return torch.tensor(0.0, device=losses.device)

# ==== 2. 模型加载与 L2 Norm (RMS) 计算 ====

print("正在加载模型...")
# 原始模型 (Base)
model_initial = AutoModelForCausalLM.from_pretrained(
    "/path/to/models/Qwen3-32B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# 微调后模型 (Warm-started / Fine-tuned)
model_final = AutoModelForCausalLM.from_pretrained(
    "/path/to/models/fft/Qwen3-32B/only_code_epoch1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("正在计算参数变化的 RMS (Root Mean Square)...")
deltas = {}
with torch.no_grad():
    for (name_initial, param_initial), (name_final, param_final) in zip(
        model_initial.named_parameters(),
        model_final.named_parameters()
    ):
        if name_initial != name_final:
            raise ValueError(f"模型结构不匹配: {name_initial} vs {name_final}")
        
        # 确保在同一设备计算
        param_final = param_final.to(param_initial.device)
        
        # [修改点 1] 使用 RMS 代替 L2 Sum
        # 公式: ||W||_RMS = ||W||_2 / sqrt(N)
        # 这样消除了矩阵大小 (Size) 对 L2 值的影响
        l2_sum = torch.norm(param_final - param_initial, p=2)
        num_elements = param_final.numel()
        delta_rms = l2_sum / np.sqrt(num_elements)
        
        deltas[name_initial] = delta_rms.item()
        
        del param_final, l2_sum, delta_rms
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==== 3. Fisher Information (Mean) 计算 ====

use_fisher = True
fisher_info = {name: 0.0 for name, _ in model_final.named_parameters()}

try:
    data_path = "/path/to/data/code/code_alpaca_20k.json"
    # 注意：如果显存不够，请将 batch_size 调小
    dataloader = create_dataloader(data_path, batch_size=2)
    
    model_final.eval()
    num_batches = 150
    
    print(f"开始计算 Fisher 信息 (Total Batches: {num_batches})...")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        input_length = batch['input_length']
        
        model_final.zero_grad()
        
        # 前向传播
        outputs = model_final(input_ids=input_ids, attention_mask=attention_mask)
        loss = sft_loss_fn(outputs.logits, input_ids, attention_mask, input_length)
        
        if loss.item() > 0:
            loss.backward()
            
            # 累积梯度平方
            for name, param in model_final.named_parameters():
                if param.grad is not None:
                    # [修改点 2] 使用 .mean() 代替 .sum()
                    # 计算的是“单位参数的平均敏感度”，让 Attention 和 MLP 在同一起跑线竞争
                    fisher_info[name] += (param.grad.detach() ** 2).mean().item()
        
        if (i + 1) % 10 == 0:
            print(f"Fisher 计算进度: {i + 1}/{num_batches}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 计算平均值 (Batch Average)
    for name in fisher_info:
        fisher_info[name] /= num_batches
    
    print("Fisher 信息计算完成。")

except Exception as e:
    print(f"⚠️ Fisher 计算失败，回退到仅使用 L2。错误信息: {e}")
    use_fisher = False

# ==== 4. 排序与筛选 (保持之前的正确逻辑) ====

all_results = []

if use_fisher:
    print("正在融合 L2 和 Fisher 分数...")
    
    # 获取参数名列表，保证索引一致
    names = list(deltas.keys())
    
    # 构造 numpy 向量 (保持原始顺序，不排序!)
    l2_vec = np.array([deltas[n] for n in names]).reshape(-1, 1)
    fisher_vec = np.array([fisher_info[n] for n in names]).reshape(-1, 1)
    
    # 归一化 (MinMaxScaler)
    # 因为已经做了 RMS/Mean 处理，这里的归一化是在“密度”层面上进行的
    scaler = MinMaxScaler()
    l2_norm = scaler.fit_transform(l2_vec).flatten()
    fisher_norm = scaler.fit_transform(fisher_vec).flatten()
    
    # 计算融合分数
    combined_scores = alpha * l2_norm + beta * fisher_norm
    
    # 打包数据
    for i, name in enumerate(names):
        all_results.append({
            "parameter": name,
            "combined_score": float(combined_scores[i]),
            "l2_delta": float(deltas[name]),    # 这里的 delta 是 RMS 值
            "fisher_info": float(fisher_info[name]), # 这里的 fisher 是 Mean 值
            "l2_norm_score": float(l2_norm[i]),
            "fisher_norm_score": float(fisher_norm[i])
        })
    
    # 根据融合后的分数统一排序
    all_results.sort(key=lambda x: x["combined_score"], reverse=True)

else:
    print("使用仅 L2 策略...")
    sorted_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    for name, delta in sorted_deltas:
        all_results.append({
            "parameter": name,
            "delta": float(delta),
            "rank": 0 
        })

# 添加 Rank 信息
for rank, item in enumerate(all_results):
    item["rank"] = rank + 1

# 截取前 10%
top_k_count = int(top_k_ratio * len(all_results))
top_k_results = all_results[:top_k_count]

# ==== 5. 保存结果 ====

# 修改文件名以区分这是 RMS/Mean 版本
full_output_file = "Qwen3-32B/newNorm/code_layer_full_ranking.json"
top_output_file = "Qwen3-32B/newNorm/code_layer_top10.json"

os.makedirs(os.path.dirname(full_output_file), exist_ok=True)

with open(full_output_file, "w") as f:
    json.dump(all_results, f, indent=4)

with open(top_output_file, "w") as f:
    json.dump(top_k_results, f, indent=4)

print("="*40)
print(f"处理完成 (方案 A: 密度归一化版)！")
print(f"总参数层数: {len(all_results)}")
print(f"选中核心层数: {len(top_k_results)}")
print(f"完整榜单已保存至: {full_output_file}")
print(f"核心参数已保存至: {top_output_file}")
print("="*40)