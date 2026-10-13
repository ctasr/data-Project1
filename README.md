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

`main.py`检查数据、固定种子42并划分训练集和验证集，记录模型比较与MLP逐轮损失。`tune_hybrid.py`运行25组融合参数，`run_formal_tuning.py`运行62组模型配置。`finalize_prediction.py`根据验证集结果选模，再读取官方测试集，用全部有标签数据重训并生成`prediction.csv`。`analyze_results.py`整理误分类和稳定性结果，`generate_figures.py`生成报告图。可选的`check_mlp_depth.py`比较MLP层数，不参与选模。

`prediction.csv`为无表头单列，每行一个0—9的类别标签。配置、指标和运行记录保存在`results/`，图保存在`figures/`。不要用`main.py --quick`生成正式结果。固定种子不能消除不同硬件或数值库造成的微小差异。
