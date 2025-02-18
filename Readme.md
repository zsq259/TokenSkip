<div align="center">
<h1><img src="assets/logo.png" height="32px"/> TokenSkip: Controllable Chain-of-Thought Compression in LLMs</h1> 
</div>

<p align="center">
<a href="https://arxiv.org">
  <img src="https://img.shields.io/badge/Arxiv-TBD-orange.svg"></a> 
<a href="https://opensource.org/licenses/Apache-2.0">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a> 
<a href="https://github.com/hemingkx/SWIFT/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-blue.svg?style=flat"></a>
</p>

## Introduction

*Does every token in the CoT output contribute equally to deriving the answer?* —— We say **NO**!

We introduce ***TokenSkip***, a simple yet effective approach that enables LLMs to selectively skip redundant tokens during Chain-of-Thought generation and learn shortcuts between critical reasoning tokens, thereby allowing for controllable CoT compression with adjustable ratios.

TokenSkip constructs compressed CoT training data with various compression ratios, by pruning unimportant tokens from original CoT trajectories. Then, it conducts a general supervised fine-tuning process on target LLMs with this training data, enabling LLMs to automatically trim redundant tokens during reasoning.

![tokenskip](./assets/tokenskip.png)

**This method is distinguished by its low training cost.** For Qwen2.5-14B-Instruct, TokenSkip fine-tunes only **0.2%** of the model's parameters using LoRA. The size of the compressed CoT training data is no larger than that of the original training set, with 7,473 examples in GSM8K and 7,500 in MATH. The training is completed in approximately **2.5 hours** for the 14B model on two 3090 GPUs. These characteristics make TokenSkip an *efficient* and *reproducible* approach, suitable for use in efficient and cost-effective LLM deployment.

We observe that as the model scale increases, there is less performance degradation at higher compression ratios, indicating that larger LLMs are better at identifying shortcuts between critical reasoning tokens, enabling more efficient CoT generation. Notably, Qwen2.5-14B-Instruct exhibits almost **NO** performance drop (less than 0.4%) with **40%** token trimming. Even at a compression ratio of 0.5, the model maintains strong reasoning capabilities, with only 2% performance degradation. 

<img src="./assets/results.png" alt="results"  />

## Todo

- [x] Release checkpoints for Qwen2.5-Instruct series
- [x] Release evaluation code on GSM8K and MATH-500
- [ ] Release code for compressed CoT data construction
- [ ] Add instructions for SFT (LoRA) on LLaMA-Factory

## Released Checkpoints

| LoRA Adapter                         | Link                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| TokenSkip-Qwen2.5-3B-Instruct-GSM8K  | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K) |
| TokenSkip-Qwen2.5-7B-Instruct-GSM8K  | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-7B-Instruct-GSM8K) |
| TokenSkip-Qwen2.5-14B-Instruct-GSM8K | [huggingface](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-14B-Instruct-GSM8K) |

## Installation

```
conda create -n tokenskip python=3.12
conda activate tokenskip
cd TokenSkip
pip install -r requirements.txt
```

## Inference

Run command lines in `eval.sh`, the results will be stored in `outputs/`.

```
./eval.sh
```

## Contributing

We warmly welcome contributions and discussions related to TokenSkip! If you have any suggestions for improvements or ideas you'd like to discuss, please don't hesitate to open an issue. This will allow us to collaborate and discuss your ideas in detail.

## Acknowledgments

This codebase is built from [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) and [LLMLingua](https://github.com/microsoft/LLMLingua).
