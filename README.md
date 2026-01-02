# DocReward\: A Document Reward Model for Structuring and Stylizing

<div align="center">

[![Paper](https://img.shields.io/badge/📄_Paper-Hugging_Face-yellow)](https://huggingface.co/papers/2510.11391)
[![Model: DocReward-3B](https://img.shields.io/badge/🤗_Model-DocReward--3B-blue)](https://huggingface.co/jeepliu/DocReward-3B)
[![Model: DocReward-7B](https://img.shields.io/badge/🤗_Model-DocReward--7B-blue)](https://huggingface.co/jeepliu/DocReward-7B)
[![Dataset: DocPair](https://img.shields.io/badge/🤗_Dataset-DocPair)](https://huggingface.co/datasets/jeepliu/DocPair)

</div>

## Introduction

Recent advances in agentic workflows have enabled the automation of tasks such as professional document generation. However, they primarily focus on textual quality, neglecting visual structure and style, which are crucial for readability and engagement. This gap arises mainly from the absence of suitable reward models to guide agentic workflows toward producing documents with stronger structural and stylistic quality. To address this, we propose **DocReward**, a document reward model that evaluates documents based on their structure and style. We construct a multi-domain dataset **DocPair** of 117K paired documents, covering 32 domains and 267 document types, each including a high- and low-professionalism document with identical content but different structure and style. This enables the model to evaluate professionalism comprehensively, and in a textual-quality-agnostic way. DocReward is trained using the Bradley-Terry loss to score documents, penalizing predictions that contradict the annotated ranking. To assess the performance of reward models, we create a test dataset containing document bundles ranked by well-educated human evaluators. Notably, DocReward outperforms GPT-4o and GPT-5 in accuracy by 30.6 and 19.4 percentage points, respectively, demonstrating its superiority over baselines. In an extrinsic evaluation of document generation, DocReward achieves a significantly higher win rate of 60.8%, compared to GPT-5's 37.7% win rate, demonstrating its utility in guiding generation agents toward producing human-preferred documents.

## Installation

### Method 1: Docker (Recommended)

```bash
bash start_docker.sh
```

> **Note:** Make sure to mount appropriate directories when running the Docker container to access your models. You can make corresponding changes in `start_docker.sh` script.

### Method 2: pip

```bash
pip install -e .
```

### Additional Requirements

After installation, ensure that the `qwen_vl_utils` package is installed:

```bash
pip install qwen_vl_utils
```

## Model Download

We provide two versions of DocReward models on Hugging Face:

- **DocReward-3B**: [https://huggingface.co/jeepliu/DocReward-3B](https://huggingface.co/jeepliu/DocReward-3B)
- **DocReward-7B**: [https://huggingface.co/jeepliu/DocReward-7B](https://huggingface.co/jeepliu/DocReward-7B)


## Demo Usage

```bash
python demo_inference.py --model_path <model path> --ckpt_dir <checkpoint dir>
```

### Parameters

- `--model_path`: Path to the model
- `--ckpt_dir`: Directory containing the checkpoint files

## Citation
If you find this work helpful, please cite out paper:
```
@misc{liu2025docrewarddocumentrewardmodel,
      title={DocReward: A Document Reward Model for Structuring and Stylizing}, 
      author={Junpeng Liu and Yuzhong Zhao and Bowen Cao and Jiayu Ding and Yilin Jia and Tengchao Lv and Yupan Huang and Shaohan Huang and Nan Yang and Li Dong and Lei Cui and Tao Ge and Xun Wang and Huitian Jiao and Sun Mao and FNU Kartik and Si-Qing Chen and Wai Lam and Furu Wei},
      year={2025},
      eprint={2510.11391},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.11391}, 
}
```

## Acknowledgements

This project is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), an excellent framework for efficient fine-tuning of large language models. We gratefully acknowledge their contribution to the open-source community.
