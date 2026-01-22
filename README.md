# LiveCodeBench Pro - LLM Benchmarking Toolkit

<p align="center">

<img width="1415" height="420" alt="image" src="https://github.com/user-attachments/assets/f5a7a439-3526-4ff4-97ce-c325d4ddc8fb" />

</p>

This repository contains a benchmarking toolkit for evaluating Large Language Models (LLMs) on competitive programming tasks. The toolkit provides a standardized way to test your LLM's code generation capabilities across a diverse set of problems.

## Overview

LiveCodeBench Pro evaluates LLMs on their ability to generate solutions for programming problems. The benchmark includes problems of varying difficulty levels from different competitive programming platforms.

## Getting Started

### Prerequisites

- Ubuntu 20.04 or higher (or other distros with kernel version >= 3.10, and cgroup support. Refer to [go-judge](https://github.com/criyle/go-judge) for more details)
- Python 3.12 or higher
- pip package manager
- docker (for running the judge server), and ensure the user has permission to run docker commands

### Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install directly using `uv`:
   ```bash
   uv sync
   ```

2. Ensure Docker is installed and running:
   ```bash
   docker --version
   ```
   
   Make sure your user has permission to run Docker commands. On Linux, you may need to add your user to the docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```
   Then log out and back in for the changes to take effect.

## How to Use

Execute the benchmark script:

```bash
python benchmark.py --model model name
```

The script will:
1. Load the LiveCodeBench-Pro dataset from Hugging Face
2. Process each problem with your LLM
3. Extract C++ code from LLM responses automatically
4. Submit solutions to the integrated judge system for evaluation
5. Collect judge results and generate comprehensive statistics
6. Save the results to `benchmark_result.json`

### (Optional) Step 4: Submit Your Results

Email your `benchmark_result.json` file to zz4242@nyu.edu to have it displayed on the leaderboard.

Please include the following information in your submission:
- LLM name and version
- Any specific details
- Contact information

## Understanding the Codebase

### api_interface.py

定义了 LLM 集成的抽象接口：
- `LLMInterface`: 抽象基类，定义了与 LLM 交互的方法
- `ExampleLLM`: 使用 OpenAI GPT-4o 的示例实现
- 提供了标准化的提示词模板，用于生成竞赛编程解决方案

### benchmark.py

主要的基准测试脚本，负责整个评估流程：
- 从 Hugging Face 加载 LiveCodeBench-Pro 数据集
- 通过 LLM 处理每个编程问题
- 自动从 LLM 响应中提取 C++ 代码
- 修复常见的 C++ 头文件问题（添加缺失的 #include）
- 将解决方案提交到集成的判题系统进行评估
- 收集判题结果并生成全面的统计数据
- 保存结果到 `benchmark_result.json`
- 支持从之前的评估中恢复（resume 功能）
- 提供多线程处理和自动重试机制

### judge.py

包含判题系统集成：
- `Judge`: 判题实现的抽象基类
- `LightCPVerifierJudge`: LightCPVerifier 集成，用于本地解决方案评估
- 自动从 Hugging Face 下载问题数据
- 支持多种编程语言（C++、Python3、PyPy3）
- 提供详细的统计信息和错误处理
- 使用 Docker 容器确保隔离的执行环境

### util.py

代码处理的实用函数：
- `extract_longest_cpp_code()`: 从 LLM 响应中智能提取 C++ 代码
- 支持多种代码格式（ fenced 代码块和自由格式）
- 智能识别包含 #include 的有效代码块

### models.yaml

LLM 模型配置文件：
- 定义了可用的语言模型及其配置
- 支持多种 API 提供商（OpenRouter、DeepSeek、GLM、Moonshot 等）
- 包含温度设置、API 端点和密钥配置
- 支持高级功能如推理模式（reasoning）和思考模式（thinking）



### Dataset

The benchmark uses the [QAQAQAQAQ/LiveCodeBench-Pro](https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro) and [QAQAQAQAQ/LiveCodeBench-Pro-Testcase](https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro-Testcase) datasets from Hugging Face, which contains competitive programming problems with varying difficulty levels.




## Contact

For questions or support, please contact us at zz4242@nyu.edu.
