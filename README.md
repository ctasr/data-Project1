# 实验一：文本分类

## 文件结构

```text
.
├─ main.py                 数据检查、固定划分、基线模型与MLP训练
├─ tune_hybrid.py          词与字符特征融合的25组参数实验
├─ run_formal_tuning.py    62组模型配置的比较实验
├─ finalize_prediction.py 验证选模、全量重训并生成测试集预测
├─ analyze_results.py     分类错误与随机种子稳定性分析
├─ generate_figures.py    生成报告所用图表
├─ check_mlp_depth.py     MLP层数对比（补充实验，不参与选模）
├─ requirements.txt       Python依赖
├─ prediction.csv         最终预测：无表头单列，每行一个0—9类别标签
├─ results/               实验配置、指标与运行记录
└─ figures/               报告图表
```

## 运行方法

环境为Python 3.11及以上版本。将课程提供的`train_data.csv`和`test_data_unlabeled.csv`放在仓库根目录；数据文件未上传GitHub。在PowerShell中依次运行：

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

正式实验按上述命令运行；`main.py --quick`仅用于调试。主实验使用随机种子42；稳定性分析另用多个种子。不同硬件或数值库可能带来微小数值差异。
