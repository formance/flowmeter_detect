# 流量计读数推理

在已加载 `flowmeter_infer:v1.1` 的 Linux Docker 宿主机运行。镜像内使用 ONNX Runtime `1.22.1` 的 CPU 执行提供程序。脚本将项目挂载到容器的 `/app`，下列输入和配置路径均相对于项目根目录。

## 模型和标定

将以下三个模型放入 `weights/` 或 `weights/onnx/`。

| 模型 | 用途 |
| --- | --- |
| `gauge_det_best.onnx` 或 `det_best.onnx` | 在原图中检测表盘。使用单类别 YOLO Detect 原始输出，脚本执行 NMS。 |
| `meter_seg_model.onnx` | 在表盘裁剪图中分割刻度线。 |
| `needle_seg_best.onnx` | 在表盘裁剪图中分割指针。 |

默认标定配置是 `src/configs/reading_080.json`。它使用 `data/scene01_seg_min190/` 中的 `080.png` 作为主模板，`039.png` 和 `058.png` 作为斜拍模板。程序将表盘透视对齐到匹配的模板，用刻度掩膜核对对齐质量，再按指针方向在主刻度之间插值。当前量程按 `RP=0` 至 `50 Nm³/h` 标定。

换表型时，通过 `--calibration` 指定对应配置。每个模板的 `reference` 相对于配置文件所在目录，`pivot` 为指针轴心的像素坐标 `[x, y]`，`anchors` 按读数从小到大列出 `[数值, x, y]`。

## 运行

```bash
cd /home/sevnce/hh/flowmeter

# 单张图片
bash scripts/infer_flowmeter.sh --source data/scene01_yolo/images/train/080.jpg

# 指定多张图片
bash scripts/infer_flowmeter.sh --source \
  data/scene01_yolo/images/test/054.jpg data/scene01_yolo/images/test/055.jpg

# 批量处理目录中的图片
bash scripts/infer_flowmeter.sh --source-dir data/scene01_yolo/images/test

# 从标准输入读取单张 Base64 图片
base64 -w0 data/scene01_yolo/images/train/080.jpg | \
  bash scripts/infer_flowmeter.sh --base64-file -

# 批量读取 Base64 图片，images.json 为 Base64 字符串数组
bash scripts/infer_flowmeter.sh --base64-file data/images.json
```

Base64 也支持 `data:image/png;base64,...` 形式。可用 `--source-base64` 直接传入字符串，或用 `--base64-file` 读取一个或多个 Base64 文本文件。

检测置信度和 NMS IoU 阈值由 `src/configs/stage1.json` 控制，默认均为 `0.5`。该文件还设置 ONNX CPU 线程数，默认 `4`。可用 `--config` 指定其他配置文件。

## 输出

默认输出目录为 `outputs/flowmeter_*/`。可用 `--output outputs/目录名` 指定一个新目录。

| 文件 | 内容 |
| --- | --- |
| `results.json` | 每张图的处理状态，以及各表盘的原图坐标框、置信度、读数 `reading`、单位 `unit`、标定模板和 `reading_status`。 |
| `image_*/overlay.png` | 原图叠加表盘框和读数，青色为指针，绿色为刻度。 |
| `image_*/dial_*_pointer.png` 和 `image_*/dial_*_scale.png` | 对应表盘裁剪区域的二值掩膜。 |

读取 `reading_status` 判断读数是否有效。对齐失败、指针或刻度掩膜缺失、读数超出标定范围等情况会返回 `reading: null`，并记录对应原因。
