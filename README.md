# TRACE: Discovering Task-Specific Parameter via Adaptation-Aware Probing for Continual Fine-Tuning

## Overview

TRACE is a novel approach to mitigate **catastrophic forgetting** in Large Language Models (LLMs) during continual fine-tuning scenarios. The core idea is to identify **task-specific core parameters** and selectively activate only these parameters during fine-tuning while freezing the remaining peripheral parameters.

### Key Features

- **Core Parameter Identification**: Automatically identifies parameters most relevant to specific tasks
- **Selective Activation**: Only activates core parameters during training, preserving general knowledge
- **Multiple Selection Methods**: Supports L2-Fisher and cosine similarity based parameter selection
- **Seamless Integration**: Built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for easy deployment



## Installation

```bash
# Clone the repository
git clone https://github.com/KEAML-JLU/TRACE.git
cd TRACE

# Install LLaMA-Factory dependencies
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Quick Start

### Step 1: Warm-Start Fine-Tuning

First, perform full fine-tuning on the target domain to obtain task-adapted weights:

```bash
llamafactory-cli train /path/to/TRACE/script/example_full.yaml
```

### Step 2: Core Parameter Selection

Choose one of the following methods to identify core parameters:

**Option A: L2-Fisher Method**
```bash
cd /path/to/TRACE/core-parameter
python selectByL2Fisher.py
```

**Option B: Cosine Similarity Method**
```bash
cd /path/to/TRACE/core-parameter
python selectBycos.py
```

The scripts will output JSON files containing the ranked parameters (e.g., `task_top_10percent.json`).

### Step 3: Freeze Fine-Tuning

Fine-tune the model with selective parameter activation:

```bash
llamafactory-cli train /path/to/TRACE/script/example_freeze.yaml
```

> **Note**: Update the core parameter file path in `LLaMA-Factory/src/llamafactory/model/adapter.py` before running.

### Step 4: Evaluation

**GSM8K (Math Reasoning)**
```bash
cd /path/to/TRACE/eval/GSM8K-eval
python main.py --model_name_or_path /path/to/your/model
```

**MedQA (Medical QA)**
```bash
cd /path/to/TRACE/eval/MedQA
python main.py
```

**HumanEval (Code Generation)**

We use [EvalPlus](https://github.com/evalplus/evalplus) for code evaluation:
```bash
pip install evalplus
evalplus.evaluate --model /path/to/your/model --dataset humaneval
```

# Citation
If you find our work useful, please cite it.

```
@inproceedings{han2026trace,
  title={TRACE: Discovering Task-Specific Parameter via Adaptation-Aware Probing for Continual Fine-Tuning},
  author={Han, Xiaosong and Chen, Ke and Dai, Xindi and Liang, Di and Peng, Minlong and Pang, Wei and Giunchiglia, Fausto and Feng, Xiaoyue and Liu, Yonghao and Guan, Renchu},
  booktitle={KDD},
  year={2026}
}
```
