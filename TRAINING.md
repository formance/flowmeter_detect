# 流量计模型训练

在 Linux 训练宿主机的项目根目录执行。训练需要 NVIDIA GPU、已配置 GPU 支持的 Docker，以及本地训练镜像。脚本默认使用 `flowmeter_train:v1.0`。使用 `flowmeter_train:v1.1` 时，在对应命令前加 `IMAGE=flowmeter_train:v1.1`。

```bash
cd /home/sevnce/hh/flowmeter
```

## 数据和初始权重

| 任务 | 数据 | 训练 / 验证 / 测试 | 初始权重 |
| --- | --- | ---: | --- |
| 表盘检测 | `data/scene01_yolo/`，包含 `data.yaml`、`images/` 和 `labels/`。类别 ID `0` 为 `meter`。 | 79 / 23 / 11 | `weights/gauge_detection_model.pt` |
| 指针分割 | `data/scene01_seg_min190/`，包含 70 张表盘裁剪图和 `xanylabel_json/` 中的指针多边形。按同名编号沿用 `data/scene01_yolo/images/` 中的划分。 | 51 / 13 / 6 | `weights/resnet18-f37072fd.pth` |

指针分割模型以 ImageNet 预训练的 ResNet-18 初始化 DeepLabV3+ 编码器。将预训练权重下载到上表指定位置。 [ResNet-18 官方权重](https://download.pytorch.org/models/resnet18-f37072fd.pth)

## 表盘检测

```bash
GPU=0 bash scripts/train_meter_det.sh
```

默认输入尺寸 `640×640`、批大小 `8`、最多 `30` 轮、早停耐心值 `10`。可通过 `--imgsz`、`--batch`、`--epochs` 和 `--patience` 调整。

结果保存在 `outputs/scene01_detect_*/`。使用其中的 `weights/best.pt` 导出检测模型，`results.csv` 保存训练与验证指标。

## 指针分割

```bash
# 单卡
GPU=0 bash scripts/train_needle_seg.sh

# 单机双卡，批大小 8 表示每卡 4 张
GPUS=0,1 bash scripts/train_needle_seg.sh --batch 8
```

默认输入尺寸 `512×512`、全局批大小 `8`、最多 `50` 轮、早停耐心值 `12`。多卡时 `--batch` 必须能被 GPU 数量整除。

结果保存在 `outputs/scene01_pointer_*/`。训练脚本按验证集指针 IoU 选出 `best.pt`，并保存 `results.csv` 和 `metrics.json`。测试指标以当前的 X-AnyLabeling 指针标注为参考。

## 导出推理权重

导出需要 `onnx==1.17.0`。如果训练镜像缺少该包，先执行一次安装命令，后续导出会挂载安装结果。

```bash
bash scripts/export_models_onnx.sh --install-onnx
```

把下面两个变量改为要导出的实际训练目录，再执行转换。也可以把两个检查点分别放到 `weights/det_best.pt` 和 `weights/needle_seg_best.pt`，直接运行脚本。

```bash
det_run=outputs/scene01_detect_YYYYMMDD_HHMMSS
needle_run=outputs/scene01_pointer_YYYYMMDD_HHMMSS_ffffff

DET_WEIGHTS="$det_run/weights/best.pt" \
NEEDLE_WEIGHTS="$needle_run/best.pt" \
bash scripts/export_models_onnx.sh
```

导出结果为 `weights/onnx/det_best.onnx` 和 `weights/onnx/needle_seg_best.onnx`。两个模型均为 FP32、固定 batch 1、opset 17，使用对应检查点的输入尺寸。部署时配合已有的 `meter_seg_model.onnx`。运行命令见 [推理说明](INFERENCE.md)。
