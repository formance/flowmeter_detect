# 流量计表盘检测与读数

面向巡检图片的流量计读数工程。三个 ONNX 模型依次检测表盘、分割指针和刻度线。程序将表盘透视对齐到标定模板，再根据指针位置计算读数。支持单张图片、批量图片和 Base64 输入。

| 用途 | Docker 镜像 | 运行设备 |
| --- | --- | --- |
| 表盘检测训练、指针分割训练 | `flowmeter_train:v1.1` | NVIDIA GPU |
| 模型导出 | `flowmeter_train:v1.1` | CPU |
| 表盘读数推理 | `flowmeter_infer:v1.1` | CPU |

两个镜像均包含 `onnx` 和 `onnxruntime`。

## 工程目录

同一任务的 Python 入口、配置和运行脚本放在一起。

| 目录 | 用途 |
| --- | --- |
| `src/training/meter_det/` | 表盘检测训练和检测模型评测 |
| `src/training/needle_seg/` | 指针分割单卡和多卡训练 |
| `src/training/export/` | 将表盘检测和指针分割模型导出为 ONNX |
| `src/inference/meter_det/` | 单张和批量表盘检测 |
| `src/inference/scale_seg/` | 刻度分割和裁剪图验证 |
| `src/inference/needle_seg/` | 指针分割和测试集可视化 |
| `src/inference/flowmeter/` | 三模型串联、表盘校正和标定读数 |
| `src/common/` | 图像输入、可视化、项目路径和共享指针模型代码 |
| `src/deploy/` | 训练与推理镜像的独立构建目录 |

数据集、模型权重和运行结果分别存放在项目根目录的 `data/`、`weights/` 和 `outputs/`。

- [训练与 ONNX 导出](docs/TRAINING.md)
- [推理、标定与输入格式](docs/INFERENCE.md)

推理使用 `gauge_det_best.onnx`、`meter_seg_model.onnx` 和 `needle_seg_best.onnx`。将权重放入 `weights/` 或 `weights/onnx/`，推理结果默认写入 `outputs/`。
