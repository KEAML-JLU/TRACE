#如果出现CUDA call was originally invoked at，需要重新下载tokenizer.json
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import time
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path
import gc
import os
from torch.nn import DataParallel
import torch.multiprocessing as mp
from contextlib import contextmanager

class MultiGPULLMEvaluator:
    def __init__(self, model_name_or_path: str, max_length: int = 2048, 
                 temperature: float = 0.1, do_sample: bool = True,
                 use_multi_gpu: bool = True, batch_size: int = 1,
                 max_memory_per_gpu: str = "40GiB"):
        """
        初始化多GPU LLM测评器
        
        Args:
            model_name_or_path: 模型路径或HuggingFace模型名
            max_length: 生成的最大长度
            temperature: 生成温度
            do_sample: 是否使用采样
            use_multi_gpu: 是否使用多GPU
            batch_size: 批处理大小
            max_memory_per_gpu: 每个GPU的最大内存限制
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_multi_gpu = use_multi_gpu
        self.batch_size = batch_size
        self.max_memory_per_gpu = max_memory_per_gpu
        
        # 设置环境变量以优化内存使用
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # 检查可用GPU
        self.device_count = torch.cuda.device_count()
        print(f"Available GPUs: {self.device_count}")
        
        # 加载模型和分词器
        self.model, self.tokenizer = self.load_model(model_name_or_path)
        
        # One-shot example
        # self.one_shot_example = {
        #     "instruction": "Based on the following clinical case, determine the cause and choose the most likely diagnosis.",
        #     "input": "A 45-year-old male presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF. Which is the most likely diagnosis?\nOptions:\nA. Anterior myocardial infarction\nB. Inferior myocardial infarction\nC. Pulmonary embolism\nD. Pneumothorax",
        #     "output": "B. Inferior myocardial infarction"
        # }
        self.few_shot_examples = [
            {
                "instruction": "Based on the following clinical case, determine the cause and choose the most likely diagnosis.",
                "input": "A 45-year-old male presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF.\nOptions:\nA. Anterior myocardial infarction\nB. Inferior myocardial infarction\nC. Pulmonary embolism\nD. Pneumothorax",
                "output": "B. Inferior myocardial infarction"
            },
            {
                "instruction": "Choose the best next step in management.",
                "input": "A 28-year-old woman with dysuria and urinary frequency. UA positive for nitrites.\nOptions:\nA. Ciprofloxacin for 3 days\nB. Nitrofurantoin for 5 days\nC. Amoxicillin for 10 days\nD. No treatment",
                "output": "B. Nitrofurantoin for 5 days"
            },
            {
                "instruction": "Select the most likely mechanism.",
                "input": "Patient with myasthenia gravis presents with fatigable ptosis.\nOptions:\nA. Presynaptic Ca2+ channel antibodies\nB. Postsynaptic ACh receptor antibodies\nC. Demyelination of motor neurons\nD. Excess acetylcholinesterase",
                "output": "B. Postsynaptic ACh receptor antibodies"
            },
            {
                "instruction": "Pick the correct epidemiology fact.",
                "input": "As disease prevalence decreases, how do PPV and NPV change?\nOptions:\nA. PPV increases, NPV decreases\nB. PPV decreases, NPV increases\nC. Both increase\nD. Both decrease",
                "output": "B. PPV decreases, NPV increases"
            },
            {
                "instruction": "Choose the most appropriate therapy.",
                "input": "A patient with panic disorder needs fast-acting abortive treatment.\nOptions:\nA. SSRI\nB. Benzodiazepine\nC. Buspirone\nD. Beta blocker",
                "output": "B. Benzodiazepine"
            }
        ]
    
    def setup_device_map(self):
        """
        设置设备映射以分配内存
        """
        if self.device_count <= 1:
            return "auto"
        
        # 为多GPU设置更精细的设备映射
        device_map = {}
        max_memory = {}
        
        for i in range(self.device_count):
            max_memory[i] = self.max_memory_per_gpu
        
        return {"device_map": "auto", "max_memory": max_memory}
    
    def load_model(self, model_name_or_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载模型和分词器，支持多GPU
        
        Args:
            model_name_or_path: 模型路径或名称
            
        Returns:
            模型和分词器的元组
        """
        print(f"Loading model from {model_name_or_path} ...")
        print(f"Using multi-GPU: {self.use_multi_gpu}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True
        )
        
        # 设置加载参数
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        if self.use_multi_gpu and self.device_count > 1:
            # 多GPU设置
            device_setup = self.setup_device_map()
            if isinstance(device_setup, dict):
                load_kwargs.update(device_setup)
            else:
                load_kwargs["device_map"] = device_setup
        else:
            # 单GPU设置
            load_kwargs["device_map"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs
        )
        
        # 设置pad_token
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0
        
        model.eval()
        
        # 打印模型信息
        print(f"Model loaded successfully!")
        if hasattr(model, 'hf_device_map'):
            print(f"Device map: {model.hf_device_map}")
        
        # 显示内存使用情况
        self.print_memory_usage()
        
        return model, tokenizer
    
    def print_memory_usage(self):
        """打印GPU内存使用情况"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    
    @contextmanager
    def memory_management(self):
        """内存管理上下文管理器"""
        try:
            yield
        finally:
            # 强制清理内存
            gc.collect()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
    
    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载测评数据集
        
        Args:
            file_path: 数据集文件路径（支持.json和.jsonl格式）
            
        Returns:
            测评数据列表
        """
        dataset = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        line = line.strip()
                        if line:
                            dataset.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        dataset = data
                    else:
                        dataset = [data]
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}")
        except json.JSONDecodeError as e:
            print(f"错误：JSON解析失败 - {e}")
        except Exception as e:
            print(f"错误：加载数据集时发生异常 - {e}")
            
        return dataset
    
    def create_prompt(self, instruction: str, input_text: str, use_one_shot: bool = True) -> str:
        """
        创建提示词（最小改动版）：
        - 若有 self.few_shot_examples，则采用最多5个示例；否则沿用 one-shot。
        - 规范输出：第一行只输出选项字母A/B/C/D；第二行起（可选）再给解释。
        """
        prompt = ""

        if use_one_shot:
            # 若类中提供 few_shot_examples，则用它的前5个；否则退回 one_shot
            examples = getattr(self, "few_shot_examples", None)
            if examples:
                prompt += "Here are some examples:\n\n"
                for idx, ex in enumerate(examples[:5], 1):
                    prompt += (
                        f"Example {idx}:\n"
                        f"Question: {ex['instruction']}\n"
                        f"{ex['input']}\n\n"
                        f"Correct Answer: {ex['output']}\n\n"
                        + "-" * 40 + "\n\n"
                    )
            else:
                example = self.one_shot_example
                prompt += "Here is an example:\n\n"
                prompt += f"Question: {example['instruction']}\n"
                prompt += f"{example['input']}\n\n"
                prompt += f"Correct Answer: {example['output']}\n\n"
                prompt += "=" * 50 + "\n\n"

        # —— 强制输出规范（先答案后解释） ——
        # prompt += (
        #     "OUTPUT FORMAT / 输出规范：\n"
        #     "1) 第一行：只输出一个大写字母（A/B/C/D），不得包含其他字符或标点；\n"
        #     "2) 第二行起（可选）：以“Explanation:”开头给出简短分析。\n\n"
        # )
        prompt += (
            "OUTPUT FORMAT:\n"
            "1) The first line: Output the answer, which is the provided option (e.g., 'A. xxx', 'B. yyy', 'C. zzz', 'D. www'); \n"
            "2) The second line and beyond (optional): Start with “Explanation:” to give a brief analysis.\n\n"
        )

        prompt += "Now please answer the following question:\n\n"
        prompt += f"Question: {instruction}\n"
        prompt += f"{input_text}\n\n"
        prompt += "Answer:"  # 紧跟着让模型先给出选项字母

        return prompt

    
    def generate_response_batch(self, prompts: List[str], max_new_tokens: int = 100) -> List[str]:
        """
        批量生成回答
        
        Args:
            prompts: 输入提示词列表
            max_new_tokens: 最大生成token数
            
        Returns:
            模型生成的回答列表
        """
        responses = []
        
        with self.memory_management():
            try:
                # 编码输入
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length - max_new_tokens,
                    padding=True
                )
                
                # 确保输入在正确的设备上
                if hasattr(self.model, 'device'):
                    device = self.model.device
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'device'):
                    device = self.model.module.device
                else:
                    device = next(self.model.parameters()).device
                
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                # 生成回答
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        do_sample=self.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                        use_cache=True,
                        return_dict_in_generate=False
                    )
                
                # 解码生成的文本
                for i, output in enumerate(outputs):
                    generated_ids = output[input_ids.shape[1]:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    responses.append(response.strip())
                
                # 清理变量
                del inputs, input_ids, attention_mask, outputs
                
            except Exception as e:
                print(f"批量生成回答时出错: {e}")
                responses = [f"ERROR: 生成失败 - {e}"] * len(prompts)
        
        return responses
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        单个生成回答（为了兼容性保留）
        
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            
        Returns:
            模型生成的回答
        """
        responses = self.generate_response_batch([prompt], max_new_tokens)
        return responses[0] if responses else "ERROR: 生成失败"
    
    def extract_answer(self, response: str) -> str:
        """
        从LLM回答中提取选项
        
        Args:
            response: LLM的完整回答
            
        Returns:
            提取的答案选项
        """
        response = response.strip()
        
        patterns = [
            r'^([A-D])\.\s*(.+?)(?:\n|$)',
            r'^([A-D]):\s*(.+?)(?:\n|$)',
            r'^([A-D])\s+(.+?)(?:\n|$)',
            r'Option\s*([A-D])[：:]\s*(.+?)(?:\n|$)',
            r'Answer\s*is\s*([A-D])[：:]\s*(.+?)(?:\n|$)',
            r'([A-D])\s*option[：:]\s*(.+?)(?:\n|$)',
            r'^([A-D])\s*[\.：:]\s*(.+?)(?:\n|$)',
        ]
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE | re.MULTILINE)
                if match:
                    option = match.group(1).upper()
                    if len(match.groups()) > 1:
                        content = match.group(2).strip()
                        return f"{option}. {content}"
                    else:
                        return option
        
        single_letter_match = re.search(r'\b([A-D])\b', response)
        if single_letter_match:
            return single_letter_match.group(1).upper()
        
        first_line = lines[0] if lines else response
        return first_line.strip()
    
    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        比较预测答案和正确答案
        
        Args:
            predicted: 预测的答案
            ground_truth: 正确答案
            
        Returns:
            是否匹配
        """
        def normalize_answer(answer: str) -> str:
            answer = answer.strip().upper()
            match = re.search(r'^([A-D])', answer)
            if match:
                return match.group(1)
            return answer
        
        pred_normalized = normalize_answer(predicted)
        truth_normalized = normalize_answer(ground_truth)
        
        return pred_normalized == truth_normalized
    
    def evaluate(self, dataset: List[Dict[str, Any]], use_one_shot: bool = True, 
                 save_results: bool = True, results_file: str = "evaluation_results.json",
                 max_new_tokens: int = 100) -> Dict[str, Any]:
        """
        执行测评，支持批处理
        
        Args:
            dataset: 测评数据集
            use_one_shot: 是否使用one-shot示例
            save_results: 是否保存详细结果
            results_file: 结果保存文件名
            max_new_tokens: 每次生成的最大token数
            
        Returns:
            测评结果统计
        """
        if not dataset:
            print("错误：数据集为空")
            return {}
        
        results = []
        correct_count = 0
        total_count = len(dataset)
        
        print(f"Starting evaluation with {total_count} samples...")
        print(f"Using model: {self.model_name_or_path}")
        print(f"Batch size: {self.batch_size}")
        print(f"One-shot example: {'Enabled' if use_one_shot else 'Disabled'}")
        print(f"Generation parameters: max_new_tokens={max_new_tokens}, temperature={self.temperature}")
        print("=" * 60)
        
        # 批处理评估
        for i in range(0, total_count, self.batch_size):
            batch_end = min(i + self.batch_size, total_count)
            batch = dataset[i:batch_end]
            batch_size_actual = len(batch)
            
            print(f"Processing batch {i//self.batch_size + 1}/{(total_count-1)//self.batch_size + 1} "
                  f"(samples {i+1}-{batch_end})...")
            
            with self.memory_management():
                try:
                    # 创建批量提示词
                    prompts = []
                    for item in batch:
                        prompt = self.create_prompt(
                            item['instruction'], 
                            item['input'], 
                            use_one_shot
                        )
                        prompts.append(prompt)
                    
                    # 批量生成回答
                    llm_responses = self.generate_response_batch(prompts, max_new_tokens)
                    
                    # 处理每个回答
                    for j, (item, llm_response) in enumerate(zip(batch, llm_responses)):
                        sample_idx = i + j
                        
                        # 提取答案
                        predicted_answer = self.extract_answer(llm_response)
                        ground_truth = item['output']
                        
                        # 比较答案
                        is_correct = self.compare_answers(predicted_answer, ground_truth)
                        if is_correct:
                            correct_count += 1
                            print(f"  Sample {sample_idx+1}: ✓")
                            print(f"    Predicted: {predicted_answer}")
                            print(f"    llm: {llm_response}")
                        else:
                            print(f"  Sample {sample_idx+1}: ✗")
                            print(f"    Predicted: {predicted_answer}")
                            print(f"    llm: {llm_response}")
                            print(f"    Correct: {ground_truth}")
                        
                        # 保存详细结果
                        result_item = {
                            "index": sample_idx,
                            "instruction": item['instruction'],
                            "input": item['input'],
                            "ground_truth": ground_truth,
                            "llm_response": llm_response,
                            "predicted_answer": predicted_answer,
                            "is_correct": is_correct,
                            "prompt": prompts[j]
                        }
                        results.append(result_item)
                    
                except Exception as e:
                    print(f"Error processing batch {i//self.batch_size + 1}: {e}")
                    # 为批次中的每个样本添加错误结果
                    for j, item in enumerate(batch):
                        sample_idx = i + j
                        result_item = {
                            "index": sample_idx,
                            "instruction": item.get('instruction', ''),
                            "input": item.get('input', ''),
                            "ground_truth": item.get('output', ''),
                            "llm_response": f"ERROR: {e}",
                            "predicted_answer": "",
                            "is_correct": False,
                            "prompt": ""
                        }
                        results.append(result_item)
            
            # 显示内存使用情况
            if (i // self.batch_size + 1) % 5 == 0:
                print("Memory usage after batch:")
                self.print_memory_usage()
        
        # 计算统计结果
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        evaluation_summary = {
            "total_samples": total_count,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "model_name": self.model_name_or_path,
            "use_one_shot": use_one_shot,
            "batch_size": self.batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": self.temperature,
            "use_multi_gpu": self.use_multi_gpu,
            "gpu_count": self.device_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果
        if save_results:
            full_results = {
                "summary": evaluation_summary,
                "detailed_results": results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed results saved to: {results_file}")
        
        # 打印总结
        print("\n" + "=" * 60)
        print("Evaluation completed!")
        print(f"Total samples: {total_count}")
        print(f"Correct count: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Batch size used: {self.batch_size}")
        print(f"GPU count: {self.device_count}")
        print("Final memory usage:")
        self.print_memory_usage()
        print("=" * 60)
        
        return evaluation_summary
    
    def __del__(self):
        """
        析构函数，清理GPU内存
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # 清理所有GPU内存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        gc.collect()

def main():
    """
    主函数 - 多GPU使用示例
    """
    # 配置参数
    MODEL_PATH = "/path/to/your/model"  # replace with your model path
    
    # 创建多GPU测评器
    evaluator = MultiGPULLMEvaluator(
        model_name_or_path=MODEL_PATH,
        max_length=2048,
        temperature=0.1,
        do_sample=True,
        use_multi_gpu=True,  # 启用多GPU
        batch_size=2,  # 根据您的GPU内存调整批处理大小
        max_memory_per_gpu="80GiB"  # 根据您的GPU内存调整
    )
    
    # 加载数据集
    dataset_path = "../data/MedQA/medqa_test_alpaca.jsonl"
    dataset = evaluator.load_dataset(dataset_path)
    
    if not dataset:
        print("Dataset file not found, using sample data for demonstration...")
        dataset = [
            {
                "instruction": "Based on the following clinical case, determine the cause and choose the most likely diagnosis.",
                "input": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\nOptions:\nA. Disclose the error to the patient and put it in the operative report\nB. Tell the attending that he cannot fail to disclose this mistake\nC. Report the physician to the ethics committee\nD. Refuse to dictate the operative report",
                "output": "B. Tell the attending that he cannot fail to disclose this mistake"
            },
            {
                "instruction": "Choose the correct answer for the following medical question.",
                "input": "A 65-year-old patient presents with chest pain. The ECG shows ST-segment elevation in leads V1-V4. What is the most likely diagnosis?\nOptions:\nA. Inferior myocardial infarction\nB. Anterior myocardial infarction\nC. Lateral myocardial infarction\nD. Posterior myocardial infarction",
                "output": "B. Anterior myocardial infarction"
            }
        ]
    
    # 执行评估
    results = evaluator.evaluate(
        dataset=dataset,
        use_one_shot=False,
        save_results=True,
        results_file="res.json",
        max_new_tokens=100
    )

if __name__ == "__main__":
    main()