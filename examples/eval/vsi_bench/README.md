# VSI-Bench Evaluation in ms-swift

[VSI-Bench](https://github.com/vision-x-nyu/thinking-in-space) (Video Spatial Intelligence Benchmark) 是一个用于评估多模态大语言模型空间推理能力的基准测试。

## 概述

VSI-Bench 包含来自 288 个第一人称视频的 5000+ 问答对，涵盖三大类任务：

1. **配置性问题 (Configurational)**: 空间布局相关问题
2. **测量估计 (Measurement Estimation)**: 数值距离/尺寸问题
3. **时空问题 (Spatiotemporal)**: 涉及时间和空间关系的问题

### 问题类型

| 类型 | 问题类型 | 答案格式 |
|------|---------|---------|
| 配置性 | object_rel_direction_hard | 多选 (A/B/C/D) |
| 配置性 | object_rel_direction_easy | 多选 (A/B/C/D) |
| 测量 | object_abs_distance | 数值 |
| 测量 | object_size_estimate | 数值 |
| 测量 | room_size_estimate | 数值 |
| 时空 | counting | 多选 |
| 时空 | appearance_order | 多选 |

### 评估指标

- **多选题 (MCA)**: 标准准确率 (Accuracy)
- **数值题 (NA)**: 平均相对准确率 (Mean Relative Accuracy, MRA)

MRA 计算公式：
```
MRA = (1/N) * Σ max(0, 1 - |pred - gt| / max(threshold, |gt|))
```

## 安装

```bash
# 安装 ms-swift
pip install ms-swift

# 安装视频处理依赖
pip install decord av moviepy
```

## 数据准备

### 下载视频/帧

VSI-Bench 使用来自 ScanNet, ScanNet++, ARKitScenes 的视频。你需要：

1. 从官方渠道获取视频数据
2. 或者使用预采样的帧

```bash
# 数据目录结构示例
vsi_bench_data/
├── arkitscenes/
│   ├── 47430468/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── 41069025/
│       └── ...
├── scannet/
│   └── ...
└── scannetpp/
    └── ...
```

## 使用方法

### 方法 1: 使用 Shell 脚本

```bash
# 编辑配置
vim eval_vsi_bench.sh

# 运行评估
bash eval_vsi_bench.sh
```

### 方法 2: 使用 Python API

```python
from swift.pipelines.eval.vsi_bench import VSIBenchEvaluator, evaluate_vsi_bench

# 简单评估
predictions = [
    {
        'id': 956,
        'question_type': 'object_abs_distance',
        'prediction': '1.8',
        'ground_truth': '1.9',
        'is_multiple_choice': False,
    },
    {
        'id': 957,
        'question_type': 'object_rel_direction_hard',
        'prediction': 'A',
        'ground_truth': 'A',
        'is_multiple_choice': True,
    },
]

results = evaluate_vsi_bench(predictions, output_path='results.json')
print(f"Overall Score: {results['overall_score']:.4f}")
```

### 方法 3: 使用命令行

```bash
# 基本用法
python -m swift.pipelines.eval.run_vsi_bench \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --video_dir /path/to/vsi_bench_data \
    --output_dir ./output

# 使用 LoRA 适配器
python -m swift.pipelines.eval.run_vsi_bench \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --adapters /path/to/lora_adapter \
    --video_dir /path/to/vsi_bench_data \
    --output_dir ./output

# 限制样本数量（用于快速测试）
python -m swift.pipelines.eval.run_vsi_bench \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --video_dir /path/to/vsi_bench_data \
    --eval_limit 100 \
    --output_dir ./output
```

### 方法 4: 使用 swift eval 命令

```bash
# 通过 swift eval 集成
swift eval \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --eval_dataset vsi-bench \
    --eval_backend Native \
    --eval_limit 100
```

## 支持的模型

理论上支持所有 ms-swift 支持的视频理解模型，包括：

- Qwen2.5-VL 系列
- InternVL2 系列
- LLaVA-Video 系列
- MiniCPM-V 系列
- CogVLM2-Video
- 等等

## 输出示例

```
============================================================
VSI-Bench Evaluation Report
============================================================

Overall Score: 0.4523

Multiple-Choice Answer (MCA):
  Accuracy: 0.5234
  Count: 2500

Numerical Answer (NA):
  MRA: 0.3812
  Count: 2500

------------------------------------------------------------
By Question Type:
------------------------------------------------------------

  object_abs_distance:
    Count: 800
    NA MRA: 0.3521 (n=800)

  object_rel_direction_hard:
    Count: 600
    MCA Accuracy: 0.4833 (n=600)

  ...

============================================================
```

## 常见问题

### Q: 如何处理没有视频只有帧的情况？

A: 将帧放在以场景名命名的目录中，脚本会自动检测并均匀采样指定数量的帧。

### Q: 采样多少帧比较合适？

A: VSI-Bench 论文推荐开源模型使用 32 帧。你可以通过 `--num_frames` 参数调整。

### Q: 如何自定义评估指标的阈值？

A: 在代码中可以调整 `mra_threshold` 参数：

```python
evaluator = VSIBenchEvaluator(mra_threshold=1.0)
```

## 参考

- [VSI-Bench 论文](https://arxiv.org/abs/2404.xxxxx)
- [VSI-Bench GitHub](https://github.com/vision-x-nyu/thinking-in-space)
- [HuggingFace 数据集](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)
