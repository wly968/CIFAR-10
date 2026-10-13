# CIFAR-10 分类（PyTorch）

阶段一：基础 CNN，测试准确率 76%。

## 运行

```bash
pip install -r requirements.txt
python cifar10.py

## 结果

| 阶段 | 方法 | 测试准确率 |
|---|---|---|
| 阶段一 | 基础 CNN | 76% |
| 最终版 | ResNet-18 + 数据增强 + CosineLR + Label Smoothing | **94.58%** |

训练配置：
- 优化器：AdamW (lr=0.001, weight_decay=5e-4)
- Epochs：100
- Batch Size：100
- 数据增强：RandomCrop、HorizontalFlip、ColorJitter、RandomErasing