# 实验一文本分类

使用Python 3.11及以上版本。把课程提供的`train_data.csv`和`test_data_unlabeled.csv`放在仓库根目录；这两个数据文件不上传GitHub。在PowerShell中依次运行：

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

文件和目录的作用：

| 文件或目录 | 作用 |
| --- | --- |
| `main.py` | 检查数据；按种子42划分训练集和验证集；比较基础模型并记录MLP逐轮损失。 |
| `tune_hybrid.py` | 比较25组词与字符融合参数。 |
| `run_formal_tuning.py` | 比较62组预设模型配置。 |
| `finalize_prediction.py` | 根据验证结果选模，用全部有标签数据重训，生成官方测试集预测。 |
| `analyze_results.py` | 整理误分类和不同随机种子的结果。 |
| `generate_figures.py` | 根据实验记录生成报告图。 |
| `check_mlp_depth.py` | 可选的MLP层数对比，不参与最终选模。 |
| `requirements.txt` | 运行所需的Python依赖。 |
| `results/`、`figures/` | 分别存放实验配置、指标与运行记录，以及报告图。 |
| `prediction.csv` | 最终预测；无表头单列，每行一个0—9的类别标签。 |

不要用`main.py --quick`生成正式结果。固定种子不能消除不同硬件或数值库造成的微小差异。
