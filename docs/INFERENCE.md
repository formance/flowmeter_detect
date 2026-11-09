# 流量计读数推理

在已加载 `flowmeter_infer:v1.1` 的 Linux Docker 宿主机运行。镜像已预装 `onnx` 和 `onnxruntime`，推理使用 `CPUExecutionProvider`。脚本默认使用该镜像，将项目按宿主机的绝对路径挂载并设为工作目录，同时提供 `/app` 挂载路径。下列输入和配置路径均相对于项目根目录。

需要重新构建推理镜像时，在项目根目录运行 `docker build -t flowmeter_infer:v1.1 src/deploy/inference`。

## 模型和标定

将以下三个模型放入 `weights/` 或 `weights/onnx/`。

| 模型 | 用途 |
| --- | --- |
| `gauge_det_best.onnx` | 在原图中检测表盘。使用单类别 YOLO Detect 原始输出，脚本执行 NMS。 |
| `meter_seg_model.onnx` | 在表盘裁剪图中分割刻度线。 |
| `needle_seg_best.onnx` | 在表盘裁剪图中分割指针。 |

默认标定配置是 `src/inference/flowmeter/reading_080.json`。模板位于 `data/scene01_seg_min190/`。场景一使用刻度较完整的 `058.png` 和适合遮挡视角的 `028.png`，场景二使用 `080.png`。程序将表盘透视对齐到匹配的模板，用刻度掩膜核对对齐质量，再从指针掩膜拟合中心线。中心线与刻度锚点折线的交点决定读数，按交点在相邻锚点间的位置插值。当前量程按 `RP=0` 至 `50 Nm³/h` 标定。

换表型时，通过 `--calibration` 指定对应配置。每个模板的 `reference` 相对于配置文件所在目录，`pivot` 为指针轴心的像素坐标 `[x, y]`，`anchors` 按读数从小到大列出 `[数值, x, y]`。

`058.png` 使用已知零流量时的指针交点校准零点。`measured_zero_fraction` 表示该交点在第一个刻度段上的位置比例，约为从 `RP` 到 `5` 的 18%。落在 `RP` 到实测零点之间的交点读作 0，实测零点到 `5` 之间重新插值。

## 运行

```bash
cd /home/sevnce/hh/flowmeter

# 单张图片
bash src/inference/flowmeter/infer_flowmeter.sh --source data/scene01_yolo/images/train/080.jpg

# 指定多张图片
bash src/inference/flowmeter/infer_flowmeter.sh --source \
  data/scene01_yolo/images/test/054.jpg data/scene01_yolo/images/test/055.jpg

# 批量处理目录中的图片
bash src/inference/flowmeter/infer_flowmeter.sh --source-dir data/scene01_yolo/images/test

# 从标准输入读取单张 Base64 图片
base64 -w0 data/scene01_yolo/images/train/080.jpg | \
  bash src/inference/flowmeter/infer_flowmeter.sh --base64-file -

# 批量读取 Base64 图片，images.json 为 Base64 字符串数组
bash src/inference/flowmeter/infer_flowmeter.sh --base64-file data/images.json
```

Base64 也支持 `data:image/png;base64,...` 形式。可用 `--source-base64` 直接传入字符串，或用 `--base64-file` 读取一个或多个 Base64 文本文件。

直接运行 Python 入口时，输入文件和图片目录的相对路径会先从当前工作目录查找，找不到时再从项目根目录查找。Docker 启动脚本支持 `--source-dir data/scene02`，也支持项目内图片的宿主机绝对路径。若项目的 `data` 是指向外部目录的软链接，脚本会将它的真实目标按相同绝对路径只读挂载到容器。

检测置信度和 NMS IoU 阈值由 `src/inference/meter_det/stage1.json` 控制，默认均为 `0.5`。该文件还设置 ONNX CPU 线程数，默认 `4`。可用 `--config` 指定其他配置文件。

## 输出

默认输出目录为 `outputs/flowmeter_*/`。可用 `--output outputs/目录名` 指定一个新目录。

| 文件 | 内容 |
| --- | --- |
| `results.json` | 每张图的处理状态、推理耗时 `inference_ms`，以及各表盘的原图坐标框、置信度、读数 `reading`、单位 `unit`、标定模板和 `reading_status`。 |
| `image_*/overlay.png` | 原图叠加表盘框和读数，青色为指针，绿色为刻度。 |
| `image_*/dial_*_pointer.png` 和 `image_*/dial_*_scale.png` | 对应表盘裁剪区域的二值掩膜。 |

读取 `reading_status` 判断读数是否有效。指针方向可信且落在零刻度以下时，返回 `reading: 0.0`。超过上限容差、对齐失败、指针或刻度掩膜缺失等情况会返回 `reading: null`，并记录对应原因。

`inference_ms` 的单位为毫秒，累计该图的表盘检测、所有表盘的指针和刻度分割、校正及读数计算时间。计时从图像解码后开始，模型和标定模板初始化、叠图绘制与文件写入在计时范围之外。未检测到表盘时只记录检测耗时，输入解码失败时返回 `null`。
