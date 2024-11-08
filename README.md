# 推荐系统实验报告

## 项目简介

本项目是一个基于矩阵分解的协同过滤推荐系统，旨在预测用户对物品的评分。项目使用了奇异值分解（SVD）技术，通过捕捉用户和物品的隐含特征来进行推荐。

### 关键特性

- **协同过滤**：利用用户的历史评分数据来推荐物品。
- **矩阵分解**：使用SVD技术处理用户-物品评分矩阵的稀疏性问题。
- **模型训练与测试**：提供了完整的模型训练和测试流程，包括数据预处理、模型初始化、参数优化等。
- **性能评估**：使用均方根误差（RMSE）来评估模型的预测准确性。

## 技术栈

- **Python**：编程语言
- **NumPy**：进行高效的数值计算
- **Pandas**：数据处理和分析
- **Matplotlib**：数据可视化
- **Pickle**：数据序列化

## 安装指南

在开始之前，请确保你已经安装了所有必要的库。你可以通过以下命令安装：

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 使用方法

### 数据准备

确保你的数据集格式与项目中的数据集格式一致。

### 运行模型

使用提供的脚本来训练和测试模型。

### 结果分析

查看模型的预测结果和性能评估指标。

#### 步骤详解

1. **数据预处理**：运行 `alloc_index.py` 和 `Get_Data.py` 来处理数据。
2. **模型训练**：使用 `SVD.py` 来训练模型。
3. **模型测试**：使用 `SVD.py` 中的测试功能来评估模型性能。

## 实验结果

实验结果表明，我们的模型在测试集上取得了良好的预测效果，RMSE值为XX，表明模型具有较高的预测准确性。

## 贡献者

- **张丛**：项目负责人，主要开发者。
- **杨征路**：指导教师。

## 许可证

本项目遵循 [MIT License](https://opensource.org/licenses/MIT)。请自由地使用和修改代码，但请保留原作者信息。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- **邮箱**：[2476003833@qq.com]
- **GitHub**：[@zciszrry](https://github.com/zciszrry)

## 致谢

感谢所有为这个项目做出贡献的人，包括我的导师和同学们。

---

## 更多信息

### 数据集说明

数据集包含用户对物品的评分信息，用于训练和测试推荐系统模型。

### 模型细节

模型使用SVD将用户-物品评分矩阵分解为用户特征矩阵和物品特征矩阵，通过优化这两个矩阵来预测未知评分。

### 实验分析

实验部分详细描述了数据预处理、模型训练、参数调优和结果评估的整个过程。

### 未来工作

- 探索更深层次的神经网络模型来改进推荐系统。
- 集成更多用户和物品的特征，以提高推荐的准确性和多样性。

### 附录

以下是项目中使用的一些关键代码片段：

```python
# SVD类定义
class SVD:
    def __init__(self, model_path, data_path, lr, lamda1, lamda2, lamda3, lamda4, factor):
        # 初始化代码
        ...

    def train(self, epochs, save, load):
        # 训练模型代码
        ...

    def predict(self, user_id, item_id):
        # 预测用户对物品的评分代码
        ...
```
