# [ACL 2026] DocReward\: A Document Reward Model for Structuring and Stylizing

<div align="center">

[![Paper](https://img.shields.io/badge/📄_Paper-Hugging_Face-yellow)](https://huggingface.co/papers/2510.11391)
[![Model: DocReward-3B](https://img.shields.io/badge/🤗_Model-DocReward--3B-blue)](https://huggingface.co/jeepliu/DocReward-3B)
[![Model: DocReward-7B](https://img.shields.io/badge/🤗_Model-DocReward--7B-blue)](https://huggingface.co/jeepliu/DocReward-7B)
[![Dataset: DocPair](https://img.shields.io/badge/🤗_Dataset-DocPair)](https://huggingface.co/datasets/jeepliu/DocPair)

</div>

## Introduction

Recent agentic workflows have automated professional document generation but focus narrowly on textual quality, overlooking structural and stylistic professionalism that is equally critical for readability. This gap stems mainly from a lack of effective reward models capable of guiding agents toward producing documents with high structural and stylistic professionalism. We introduce DocReward, a Document Reward Model that evaluates documents based on their structure and style. To achieve this, we propose a textual-quality-agnostic framework that ensures assessments are not confounded by content quality, and construct DocPair, a dataset of 117K paired documents, covering 32 domains and 267 types. DocReward is trained using the Bradley-Terry loss. On a manually annotated benchmark, DocReward outperforms GPT-5 by 14.6 percentage points in accuracy. Reinforcement learning experiments further show that DocReward effectively guides agents toward generating documents of greater structural and stylistic quality. 

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
