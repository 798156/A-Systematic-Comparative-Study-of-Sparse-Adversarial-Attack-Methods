# A Systematic Comparative Study of Sparse Adversarial Attack Methods

## 项目概述

本项目系统性地比较研究了L0稀疏对抗攻击方法，通过统一实验设置评估了五种代表性方法（梯度法、几何法、启发式法），为深度学习安全领域的研究人员提供标准化的基准测试。

## 研究背景

深度神经网络在图像分类、目标检测等任务中表现出色，但对对抗攻击极其脆弱。L0稀疏攻击通过修改极少量的像素点就能误导模型，对安全敏感应用构成严重威胁。然而，现有L0攻击方法缺乏系统性比较，研究人员难以选择合适的方法。

## 核心贡献

### 1. 统一基准测试框架
- 标准化实验设置，消除方法间差异
- 覆盖攻击效果、效率、稳定性、防御鲁棒性四个维度
- 1500+对抗样本的全面评估

### 2. 五种代表性方法比较
- **梯度法**: JSMA、PixelGrad
- **几何法**: SparseFool  
- **启发式法**: Greedy、RandomSparse

### 3. 关键发现
- **Greedy方法**在实时攻击场景具有理论可行性（30fps视频流攻击）
- **JSMA+Greedy组合**可实现88.7%的联合攻击成功率
- **SparseFool**对L∞防御最具鲁棒性

## 代码结构详解

### 核心文件架构
```
SparseAttackRL/
├── 配置文件
│   ├── config.yaml                 # 统一配置参数（攻击参数、模型配置、评估指标）
│   └── requirements.txt            # 依赖库清单
│
├── 主要实验文件
│   ├── unified_baseline_test.py    # 统一基线测试（主实验入口）
│   ├── analyze_all_5methods.py     # 五种方法全面对比分析
│   ├── analyze_failure_cases.py    # 失败案例深度分析
│   ├── analyze_query_efficiency.py # 查询效率分析
│   └── statistical_analysis.py     # 统计显著性检验
│
├── 攻击方法实现
│   ├── jsma_attack.py              # JSMA梯度攻击
│   ├── pixel_gradient_attack.py    # PixelGrad梯度攻击
│   ├── sparsefool_attack.py        # SparseFool几何攻击
│   ├── greedy_attack.py            # Greedy启发式攻击
│   ├── random_sparse_attack.py     # RandomSparse基线攻击
│   └── hybrid_attack.py            # 混合攻击策略
│
├── 模型相关
│   ├── target_model.py             # 目标模型定义（ResNet18/VGG16/MobileNetV2）
│   ├── model_loader.py             # 模型加载工具
│   ├── train_cifar10_*.py          # 模型训练脚本（多个版本）
│   └── load_trained_model.py       # 预训练模型加载
│
├── RL训练模块
│   ├── sparse_attack_env.py        # 稀疏攻击环境定义
│   ├── sparse_attack_env_v2.py     # 环境v2版本
│   ├── ppo_trainer*.py             # PPO训练器（多个版本）
│   └── train_resnet18_rl_v3.py     # RL模型训练
│
├── 评估与可视化
│   ├── evaluation_metrics.py       # 评估指标计算
│   ├── visualization.py            # 结果可视化
│   ├── compare_all_results.py      # 结果对比分析
│   └── display_correct_results.py  # 正确结果展示
│
├── 调试与诊断
│   ├── debug_*.py                  # 各类调试脚本
│   ├── diagnose_*.py               # 诊断工具
│   └── test_*.py                   # 测试脚本
│
├── 数据与结果
│   ├── data/                       # CIFAR-10数据集
│   ├── results/                    # 实验结果存储
│   │   ├── analysis_5methods/      # 五种方法分析结果
│   │   ├── failure_analysis/        # 失败案例分析
│   │   ├── query_efficiency/        # 查询效率结果
│   │   └── defended_model/          # 防御模型结果
│   ├── models/                     # 预训练模型存储
│   └── logs/                       # 训练日志
│
└── 论文与文档
    ├── latex_paper/                # LaTeX论文源码
    ├── paper_materials/            # 论文材料（图表、数据）
    └── *.md                        # 各类总结文档
```

### 核心逻辑流程

#### 1. 实验主流程（unified_baseline_test.py）
```
加载配置 → 初始化模型 → 选择攻击方法 → 执行攻击 → 评估结果 → 保存数据
```

#### 2. 攻击方法调用流程
```
攻击函数 → 参数解析 → 目标模型预测 → 对抗样本生成 → 成功率计算
```

#### 3. 结果分析流程
```
加载实验数据 → 统计计算 → 可视化生成 → 对比分析 → 生成报告
```

## 快速开始

### 环境配置
```bash
# 克隆项目
git clone <repository-url>
cd A_Systematic_Comparative_Study_of_Sparse_Adversarial_Attack_Methods

# 安装依赖
pip install -r requirements.txt

# 验证环境
python check_gpu.py
```

### 运行实验
```bash
# 1. 运行完整基线测试
python unified_baseline_test.py

# 2. 分析五种攻击方法
python analyze_all_5methods.py

# 3. 研究失败案例
python analyze_failure_cases.py

# 4. 查询效率分析
python analyze_query_efficiency.py
```

### 自定义实验
```bash
# 修改配置文件
vim config.yaml

# 运行特定攻击方法测试
python final_baseline_test.py --attack_method jsma

# 验证模型准确性
python check_model_accuracy.py
```

## 关键结果解读

### 方法性能对比表
| 方法 | 成功率 | 查询数 | 执行时间 | 稳定性 | 防御鲁棒性 |
|------|--------|--------|----------|--------|------------|
| JSMA | 94.2% | 28 | 0.312s | 中等 | 低 |
| PixelGrad | 76.8% | 45 | 0.156s | 高 | 中等 |
| SparseFool | 89.7% | 35 | 0.198s | 中等 | 高 |
| Greedy | 92.9% | 38 | 0.082s | 中等 | 中等 |

### 实用选择建议
- **实时攻击场景** → Greedy（最快执行速度）
- **黑盒攻击场景** → SparseFool（最强防御鲁棒性）
- **高稳定性要求** → PixelGrad（最低参数敏感性）
- **互补攻击策略** → JSMA + Greedy组合（88.7%成功率）

## 高级功能

### 1. 参数敏感性分析
```bash
# 运行参数扫描
python ablation_study.py

# 分析不同参数组合的影响
python analyze_all_5methods.py --param_sensitivity
```

### 2. 防御模型测试
```bash
# 测试对抗防御
python test_on_defended_model.py

# 多防御模型对比
python compare_all_results.py --defense_models
```

### 3. 统计显著性检验
```bash
# 运行统计检验
python statistical_analysis.py

# 生成置信区间
python statistical_analysis.py --confidence_intervals
```

##  如何贡献

1. **新增攻击方法** - 在attack_adapters.py中添加新攻击
2. **扩展评估指标** - 在evaluation_metrics.py中定义新指标
3. **改进可视化** - 在visualization.py中添加新图表
4. **优化性能** - 优化核心算法实现

## 使用须知

本研究仅限学术和安全研究用途，禁止用于任何恶意攻击活动。请遵守：
-  负责任的披露原则
-  学术诚信规范
-  道德伦理标准
-  禁止商业恶意用途

##  联系方式

如有问题或合作意向，请联系：[1412118291@qq.com]
---

**亮点：**
-  首个L0稀疏攻击标准化基准
-  1500+样本全面评估
-  实时攻击理论可行性验证
-  多维度系统性比较框架
-  防御鲁棒性深度分析
