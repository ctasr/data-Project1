# 实验一文本分类

## 运行

将课程提供的`train_data.csv`和`test_data_unlabeled.csv`放在本目录，然后依次运行：

```bash
pip install -r requirements.txt
python main.py
python tune_hybrid.py
python run_formal_tuning.py
python generate_figures.py
```

`main.py`最终生成无表头的`prediction.csv`，每行一个预测标签。

## 文件说明

| 文件 | 作用 |
|---|---|
| `main.py` | 完成数据检查、训练验证划分、模型比较、最终训练与预测 |
| `tune_hybrid.py` | 完成词字符融合模型的25组联合调参 |
| `run_formal_tuning.py` | 完成四类模型的62组正式调参 |
| `generate_figures.py` | 根据实验结果生成报告图表 |
| `requirements.txt` | 记录运行所需的Python库及版本 |
| `results/` | 保存调参结果、损失记录和分类结果 |
| `figures/` | 保存报告使用的三张实验图 |
