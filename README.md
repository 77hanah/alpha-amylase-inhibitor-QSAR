# 复现：《利用化学描述符和机器学习对作为α-淀粉酶抑制剂的吡啶酮衍生物进行QSAR建模》

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## 📖 项目简介 (About The Project)

本项目是对发表于 *ChemistrySelect* 上的研究论文 **"QSAR Modeling of Pyridone Derivatives as α-amylase Inhibitors Using Chemical Descriptors and Machine Learning"** (DOI: `10.1002/slct.202404214`) 的完整计算流程复现。

该研究旨在通过化学描述符和机器学习技术，系统性地探索32种吡啶酮衍生物的化学结构与其对α-淀粉酶抑制活性之间的关系，从而构建一个稳健且具有预测能力的定量构效关系（QSAR）模型，以支持抗糖尿病药物的发现，本项目的复现工作覆盖了从QSAR模型构建到分子对接验证的全部关键步骤。

1.  **QSAR 建模 (Python):本项目基于 Python 生态完整复现了上述研究的核心流程，相比原文献所用软件工具，全程采用 Python 开源库实现可复现、可扩展的分析流程，核心环节如下：**
    *   **数据准备**: 从SMILES表达式开始。
    *   **化学描述符计算**: 替代 PaDEL-Descriptor 软件，通过 RDKit 库批量计算 200 + 分子描述符。
    *   **数据预处理**: 基于 Scikit-learn 实现归一化及数据集划分。
    *   **模型构建与优化**:
        *   利用 PCA（主成分分析）完成特征降维，结合 Matplotlib/SciencePlots 实现得分图、贡献率雷达图、特征载荷图可视化。
        *   构建前馈神经网络 (Feedforward Neural Network) 回归模型。
        *   使用网格搜索算法 (Grid Search Algorithm) 优化神经网络的隐藏层神经元数量。
    *   **模型验证**:
        *   蒙特卡洛模型适用域评估 (Monte Carlo Model Applicability Domain Evaluation)。
        *   Y-随机化测试 (Y-randomization Test)。

2.  **设计新分子**
    *   保持吡啶酮骨架，在zinc 15数据库中基于PCA分析选择相似片段，设计了23种新化合物，并预测其活性。

3.  **SwissADME 特征预测**
    *   评估高活性候选化合物的ADME特性，筛选出具有良好ADME特性的候选分子。
  
4.  **分子对接 (AutoDock Vina & Discovery Studio)**:
    *   对原文中提出的高活性候选化合物 (N1, N3, N11) 进行准备。
    *   将其与靶蛋白 α-淀粉酶 (PDB: 3BAJ) 进行分子对接，评估结合亲和力。
    *   可视化并分析对接构象，验证其与活性位点关键氨基酸的相互作用。

## 🛠️ 技术栈 (Built With)

*   **QSAR 建模 & 数据分析**:
    *   Python 3.13.5
    *   NumPy, Pandas, Scikit-learn
    *   Matplotlib, Seaborn

*   **化学信息学 & 分子对接**:
    *   RDKit  (描述符计算)
    *   SwissADME (ADME特征预测)
    *   AutoDock Vina (分子对接)
    *   PyMOL(重对接可视化)
    *   Discovery Studio Visualizer (结果可视化)
    *   

## 🚀 项目结构 (Project Structure)

```
.
├── autodock/
│   └── conf_*.txt                # AutoDock Vina 配置文件
├── data/
│   ├── SMILES.csv                # 原始数据文件，包含化合物的 SMILES 表达式和活性数据
├── results/
│   ├── figures/                  # 保存结果图表
│   └── outputs/                  # 保存数据结果以及模型输出文件
├── src/
│   ├── applicability_domain_evaluation.py  # 蒙特卡洛模型适用域评估
│   ├── calculate_descriptors.py  # 计算化学描述符
│   ├── config.py                 # 配置文件，包含路径和参数设置
│   ├── data_process.py           # 数据预处理
│   ├── main.py                   # 主运行脚本，整合所有步骤  
│   ├── model_results.py          # 模型训练与评估
│   ├── PCA.py                    # PCA分析与可视化
│   ├── predict_new_compounds.py  # 新化合物活性预测
│   └── y_randomation_evaluate.py # Y-随机化测试
├── requirements.txt              # 项目依赖
└── README.md
```

## ⚙️ 如何运行 (Usage)

本项目包含两个主要部分：QSAR建模和分子对接。请按顺序执行。

### Part 1: QSAR 建模 (Python)

1.  **环境配置**:
    ```bash
    # 克隆仓库
    git clone https://github.com/Liu/alpha-amylase-inhibitor-QSAR.git
    cd alpha-amylase-inhibitor-QSAR

    # 创建并激活虚拟环境
    python -m venv venv
    source venv/bin/activate

    # 安装Python依赖
    pip install -r requirements.txt
    ```

2.  **运行QSAR流程**:
    配置完成后，直接运行 `main.py` 即可启动从数据预处理到模型评估的完整流程。
    ```bash
    python src/main.py
    ```
    脚本执行完毕后，所有结果（包括模型文件、性能指标和可视化图表）将自动保存到 `results/` 目录下。

### Part 2: 分子对接 (AutoDock Vina & PyMOL/Discovery Studio)

这部分操作依赖多个外部软件，展示了从蛋白准备到结果可视化的完整流程。相关文件均存放于 `autodock/` 目录下。

#### 1. 软件准备

*   **ADME预测**: [SwissADME](https://www.swiss-adme.ch/)
*   **对接引擎**: [AutoDock Vina](http://vina.scripps.edu/download.html)
*   **文件准备**: [MGLTools](http://mgltools.scripps.edu/downloads) 或 [Open Babel](https://open-babel.readthedocs.io/en/latest/index.html)
*   **结构优化与可视化**: [PyMOL](https://pymol.org/2/) 或 [Discovery Studio Visualizer (DSV)](https://www.3ds.com/products-services/biovia/products/molecular-modeling-simulation/biovia-discovery-studio/visualization/)

#### 2. 工作流程概述

**a. 新设计化合物的活性预测**
    直接运行 `main.py` 脚本后，预测结果将保存在 `results/outputs/new_compounds_predicted.csv` 中。
    ```bash
    python src/predict_new.py
    ```

**b.ADME特性预测**
    访问 [SwissADME](https://www.swiss-adme.ch/)，输入高活性候选化合物的SMILES字符串，获取其ADME特性评估结果。

**c. 受体与配体的准备**

*   **受体 (Receptor)**: 从PDB数据库获取α-淀粉酶结构 (`PDB ID: 3BAJ`)。使用PyMOL分离蛋白链，加氢后保存为 `receptor.pdb`。随后将其转换为 `receptor.pdbqt` 格式。

*   **候选配体 (Ligands N1, N3, N11)**: 化合物的SMILES式通过在线工具生成3D结构(.sdf)，使用Avogadro进行能量最小化后保存为 `.pdb`。最后使用Open Babel转换为 `.pdbqt` 格式，并计算Gasteiger电荷。
    ```bash
    # 示例命令
    obabel ligand_N11.pdb -O ligand_N11.pdbqt --partialcharge gasteiger
    ```
*   **原始配体 (Native Ligand)**: 从 `3BAJ` 晶体结构中提取原始配体，按上述类似流程处理为 `native_ligand.pdbqt`。

**d. 定义对接盒子 & 创建配置文件**

*   采用**共晶结构法**，以原始配体在晶体结构中的位置为中心，定义对接盒子（Grid Box）的中心和尺寸。
*   为每个对接任务创建独立的Vina配置文件 (`conf_native.txt`, `conf_*.txt` 等)，指定受体、配体和对接盒子参数。这些文件已保存在 `autodock/` 中。

**e. 运行Vina对接**

在`autodock/`目录下，通过命令行调用Vina执行对接。对接得分和日志将输出到 `.txt` 文件。

```bash
# 示例：运行原始配体的重对接验证
vina --config autodock/conf_native.txt --out autodock/output_native.pdbqt > autodock/log_native.txt
```
**f. 结果分析与可视化**

**重对接验证 (Redocking)**: 在PyMOL中加载受体、原始配体和重对接后的配体构象。通过`align`命令计算RMSD值（本文献复现值为1.388 Å），并渲染生成对比图。
**分子对接结果图**: 使用Discovery Studio Visualizer (DSV) 或PyMOL，分析对接后配体与受体活性位点残基的相互作用（如氢键、疏水作用等），并生成2D和3D相互作用图。


## 📊 复现结果 (Results)

### QSAR 模型性能

**1. 神经网络模型拟合结果**
*   训练集 R: `0.8551` (原文: 0.79999)
*   验证集 R: `0.9225` (原文: 0.88458)
*   测试集 R: `0.8279` (原文: 0.86475)
*   整体数据 R: `0.8180` (原文: 0.74536)

具体图表见 results\figures\regression_performance.png

**2. 测试集预测结果与实验值对比**

具体图表见 results\figures\actual_vs_predicted.png

**3. 测试集预测结果**

具体图表见 results\figures\test_predictions_table.png

### 分子对接结果

本项目复现了候选化合物与靶蛋白活性位点的对接。

**对接得分 (Binding Affinity)**:
| 化合物 | 复现得分 (kcal/mol) | 原文得分 (kcal/mol) |
| :----: | :------------------: | :------------------: |
|   N1   |       `-8.506`       |         -9.2         |
|   N3   |       `-8.685`       |         -9.1         |
|   N11  |       `-8.010`       |         -8.0         |

**分子对接结果图 (2D、3D)**:

展示了化合物与靶蛋白活性位点的相互作用，包括氢键、疏水作用等。


## 📄 引用 (Citation)

本项目的复现基于以下原始研究：
```
Zhang, Y.-K., Tong, J.-B., Chang, Z.-L., Yan, J., Xing, X.-Y., Yang, Y.-L., & Xue, Z. (2025). QSAR Modeling of Pyridone Derivatives as α-amylase Inhibitors Using Chemical Descriptors and Machine Learning. ChemistrySelect, 10(4), e202404214. https://doi.org/10.1002/slct.202404214
```

## 📧 联系方式 (Contact)

Liu - 18394599982@163.com

项目链接: [https://github.com/77hanah/alpha-amylase-inhibitor-QSAR](https://github.com/77hanah/alpha-amylase-inhibitor-QSAR)