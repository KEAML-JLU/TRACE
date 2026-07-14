import json

# 输入输出路径
input_path = "/path/to/LLaMA-Factory/data/MedQA/phrases_no_exclude_train.jsonl"
output_path = "/path/to/LLaMA-Factory/data/MedQA/medqa_train_alpaca.jsonl"

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        example = json.loads(line.strip())

        question = example.get("question", "")
        answer = example.get("answer", "")
        options = example.get("options", {})
        correct_option = example.get("answer_idx", "")

        # 构建 input 部分
        input_text = question.strip()
        if options:
            input_text += "\n选项：\n" + "\n".join([f"{k}. {v}" for k, v in options.items()])

        # 构建 output
        output_text = f"{correct_option}. {options.get(correct_option, answer)}"

        # 组装 Alpaca 格式
        alpaca_example = {
            "instruction": "Based on the following clinical case, determine the cause and choose the most likely diagnosis.",
            "input": input_text,
            "output": output_text
        }

        # 写入一行 JSONL
        outfile.write(json.dumps(alpaca_example, ensure_ascii=False) + "\n")

print(f"✅ 转换完成！文件保存在：{output_path}")
