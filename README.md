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

快速测试可把 `--samples` 改成 `1000`，把 `--epochs` 改成 `2`。训练指标和示例翻译会直接打印在终端。

## 3. 参数说明

- `--dataset`：选择 `cmn` 英中或 `fra` 英法。
- `--algorithm`：选择 `original` 原算法或 `improved` 优化算法。
- `--samples`：使用多少条句对。
- `--epochs`：训练轮数。
- `--latent-dim`：LSTM 隐状态维度。
- `--embedding-dim`：优化算法使用的字符嵌入维度。
- `--show-summary`：打印模型结构。

## 4. 实验报告

报告已写在 `report.md`，本次短实验使用 1000 条样本、8 轮训练。若要让翻译质量更明显，可以把 `--samples` 提高到 `10000`，把 `--epochs` 提高到 `30` 或更多。
