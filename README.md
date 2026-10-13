# 实验一文本分类

## 运行环境与数据

使用Python 3.11及以上版本。将课程提供的`train_data.csv`和`test_data_unlabeled.csv`放在本目录；数据文件不上传GitHub。建议从干净目录和独立环境运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe tune_hybrid.py
.\.venv\Scripts\python.exe run_formal_tuning.py
.\.venv\Scripts\python.exe finalize_prediction.py
.\.venv\Scripts\python.exe analyze_results.py
.\.venv\Scripts\python.exe generate_figures.py
```

`main.py`只使用有标签数据，固定种子42、分层划分训练集与验证集，并保存预处理前后审计、逐轮损失、模型比较和划分索引。`tune_hybrid.py`运行25组联合参数；`run_formal_tuning.py`从空检查点运行62组正式配置，逐组保存。如果中途停止，使用`python run_formal_tuning.py --resume`继续。只有这两项调参完成后，`finalize_prediction.py`才按验证集Accuracy、Macro-F1的预定顺序选模、核对最佳配置的验证结果、读取官方测试集并用全部有标签数据重拟合。

最终输出是无表头、单列、每行一个0—9标签的`prediction.csv`。`analyze_results.py`在选模后记录误分类、固定配置的三次划分复核和匹配特征预算的消融；它们不改变最终选模决定。`results/`保存完整配置、指标、环境与数据哈希、损失记录和分类结果；`figures/`保存报告图。固定种子可复现本机运行条件下的划分与随机训练过程，但不同硬件或底层数值库仍可能产生微小浮点差异。不要使用`main.py --quick`的输出作为正式结果。

`analyze_results.py`还在相同划分、词范围和LinearSVC参数下比较词袋计数与词级TF-IDF；该补充实验仅用于理解表示方法，不会改变既定的选模结果。

可选的`python check_mlp_depth.py`只回答讲义中的网络层数扩展问题，不参与选模，也不会改写`prediction.csv`。其结果保存在`results/mlp_depth_comparison.csv`和`results/mlp_depth_history.csv`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `main.py` | 数据检查、固定划分、基线比较、损失和类别指标 |
| `tune_hybrid.py` | 词字符融合的25组参数组合 |
| `run_formal_tuning.py` | 朴素贝叶斯、逻辑回归、SVC和MLP的62组正式配置 |
| `finalize_prediction.py` | 选模核验、全部有标签数据重拟合、生成预测CSV |
| `analyze_results.py` | 误分类记录和最终配置的三次划分复核 |
| `check_mlp_depth.py` | 固定其他条件，比较单隐层与双隐层MLP |
| `generate_figures.py` | 根据实测结果生成报告图 |
| `requirements.txt` | 精确的软件依赖版本 |
