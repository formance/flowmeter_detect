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

默认标定配置是 `src/inference/flowmeter/reading_080.json`。10 张已确认的模板图位于同目录的 `templates/`，覆盖不同表盘、拍摄角度和参考读数。程序将表盘透视对齐到匹配的模板，用刻度掩膜核对对齐质量，再从指针掩膜拟合中心线。中心线与刻度锚点折线的交点决定读数，按交点在相邻锚点间的位置插值。当前量程按 `RP=0` 至 `50 Nm³/h` 标定。

| 优先级 `priority` | 模板 | 用途 |
| --- | --- | --- |
| 0 | `scene02_003`、`scene03_037`、`scene03_032`、`scene03_007`、`scene03_017`、`scene01_081`、`scene02_005` | 7 张无遮挡模板，兼顾清晰度、参考读数及视角 |
| 1 | `scene02_007`、`scene02_017` | 上端刻度附近有遮挡，作为补充视角 |
| 2 | `scene02_001` | 零点附近有遮挡，降低匹配权重 |

换表型时，通过 `--calibration` 指定对应配置。每个模板的 `reference` 相对于配置文件所在目录，`pivot` 为指针轴心的像素坐标 `[x, y]`，`anchors` 按读数从小到大列出 `[数值, x, y]`。

配准优先使用模板的 `alignment_mask` 中的静态表盘特征，排除指针及表盘外部区域。所有模板同时比较原尺寸和长边 320 像素的配准结果。有效候选按刻度匹配分数、内点数量、内点比例和重投影误差综合评分；重投影误差按模板刻度半径归一化。优先级 0、1、2 的评分权重分别为 1、0.8、0.6，因此遮挡模板在几何匹配明显更好时仍可被选中。

静态特征候选均无法读数时，回退到全图特征。仍无法读数时，程序对表盘做灰度降噪和局部对比度增强，重新分割刻度，并与同样增强过的模板配准。坐标变换会换算回原始裁剪图，保存的刻度掩膜与实际采用的读数结果对应。

`known_reading` 记录参考图的已知读数。`reading_calibration` 是每张模板的分段线性校准曲线，每项为 `[交点原始读数, 校准读数]`。程序先计算未舍入的指针交点读数，再做插值校准，最后保留一位小数；未配置时使用恒等映射。曲线必须保持量程端点，原始值严格递增，校准值单调不减。`calibration_samples` 记录校准来源，仅用于复核，不参与运行时按图片名称判断读数。

当前曲线使用 10 张已确认模板的实测值，并补充同一表盘的 `scene01/058.jpg`、`scene01/039.jpg` 和 `scene02/016.jpg`，修正零点及局部系统偏差。统计未参与数值标定的图片误差时，应排除这 13 张校准图及其完全重复图；本次方案也参考了这些数据集的整体误差，因此剩余图片不属于独立盲测集，不能据此推断新表型或新场景的精度。修改模板图、轴心、锚点或分割模型后，需要重新核对配准掩膜与校准曲线。

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
| `results.json` | 每张图的处理状态、推理耗时 `inference_ms`，以及各表盘的原图坐标框、置信度、校准读数 `reading`、校准前读数 `raw_reading`、单位 `unit`、标定模板和 `reading_status`。`template_priority` 记录模板优先级。`alignment_features` 为 `static` 或 `all`，`alignment_inlier_ratio` 记录内点比例，`alignment_reprojection_error` 为模板像素单位的重投影均方根误差，`alignment_error_relative` 为误差与刻度半径之比。配准方式 `alignment_mode` 为 `native`、`resize_320` 或 `contrast_resize_320`，刻度预处理 `scale_preprocessing` 为 `original` 或 `denoise_clahe`。 |
| `image_*/overlay.png` | 原图叠加表盘框和读数，青色为指针，绿色为刻度。 |
| `image_*/dial_*_pointer.png` 和 `image_*/dial_*_scale.png` | 对应表盘裁剪区域的二值掩膜。 |

读取 `reading_status` 判断读数是否有效。指针方向可信且落在零刻度以下时，返回 `reading: 0.0`。超过上限容差、对齐失败、指针或刻度掩膜缺失等情况会返回 `reading: null`，并记录对应原因。

`inference_ms` 的单位为毫秒，累计该图的表盘检测、所有表盘的指针和刻度分割、校正及读数计算时间。计时从图像解码后开始，模型和标定模板初始化、叠图绘制与文件写入在计时范围之外。未检测到表盘时只记录检测耗时，输入解码失败时返回 `null`。
