📁 ME2-BERT 复现实验结果

本目录包含本项目在 MoralEvents 数据集上的复现实验结果，包括训练曲线、超参数设置、模型性能指标以及训练日志。所有结果均基于 bert-base-uncased，训练样本约 3000 条。

### 🟦 1. Loss Curve（训练损失曲线）

文件：loss_curve.png
说明：展示整个训练过程中主任务与辅助任务的损失变化趋势。用于判断模型是否收敛、是否出现过拟合。

🟦 2. α vs Margin（对抗系数与间隔调度）

文件：alpha_margin.png / margin_schedule.png
说明：反映 GRL 对抗系数 α 随迭代逐渐增大，以及 margin 随 epoch 衰减的情况。是 ME2-BERT 多任务/对抗训练的重要调度机制。

🟦 3. F1 Score Comparison（关键结果）

文件：f1_table_full_vs_baseline.png
说明：展示 Full model（含 E-DAE）与 Baseline（无 E-DAE）的 F1-Macro / 各标签 F1 对比。

