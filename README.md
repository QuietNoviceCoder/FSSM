# FS4D: Feedback State Space Model

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

FS4D 是一个基于状态空间模型（State Space Models）的研究项目，通过引入**跨层反馈机制**和**隐式定点计算**（Implicit Fixed-Point Computation），旨在增强模型对超长序列的建模能力。

## 🚀 核心特性

- **反馈状态空间架构 (FS4D)**：在传统 SSM（如 S4, Mamba）基础上，引入反馈路径 $f$，实现状态的动态自适应。
- **隐式定点迭代**：利用隐式函数理论，通过迭代寻优实现反馈项的定点计算。
- **高性能表现**：在 Long Range Arena (LRA) 测试中表现出色，尤其在 **Pathfinder** 任务上达到了 **95.60%** 的准确率。

## 📁 项目结构

```text
├── lra/                # Long Range Arena 任务代码 (Pathfinder, ListOps, etc.)
├── fssm.py            # FS4D 核心模型定义
├── SSM_function.py     # SSM 基础算子与数学变换函数
├── LICENSE             # BSD-2-Clause 许可证
└── README.md

