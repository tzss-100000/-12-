# 语音翻译实验

本实验复现原始字符级 Seq2Seq 翻译模型，并提供一个优化版：Embedding + 双向 Encoder LSTM + 稀疏交叉熵。数据集会自动下载到 `data/`，不需要提前准备本地数据。

## 1. 创建环境

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

## 2. 运行实验

优化算法，英中数据集：

```powershell
python speech_translation_experiment.py --dataset cmn --algorithm improved --samples 20000 --epochs 50
```

原算法，英中数据集：

```powershell
python speech_translation_experiment.py --dataset cmn --algorithm original --samples 20000 --epochs 50
```

优化算法，英法数据集：

```powershell
python speech_translation_experiment.py --dataset fra --algorithm improved --samples 20000 --epochs 50
```

原算法，英法数据集：

```powershell
python speech_translation_experiment.py --dataset fra --algorithm original --samples 20000 --epochs 50
```
