# DeepLabV3+ 指针和刻度联合分割

## 数据集

已准备 data/pointer_scale_seg/：scene01_part 64 张、scene02 26 张、scene03 34 张，共 124 张人工确认样本。回收站的 12 张图片未使用。原始标注工具的数据、原图和 WSL 项目均未修改。

```text
data/pointer_scale_seg/
├── images/{train,val,test}/
├── masks/{train,val,test}/
├── annotations/{train,val,test}/
├── splits/{train,val,test}.txt
├── by_scene/{scene01_part,scene02,scene03}/  # 三个单独划分的数据集
├── classes.json
└── manifest.json
```

每个场景分别采用 7:2:1，使用最大余数法取整：45/13/6、18/5/3、24/7/3，合并为 **87/25/12**。图片编号、场景和来源均保留。

标签是单通道 uint8 PNG，像素值 **0 background、1 pointer、2 scale_tick**。指针多边形直接栅格化，刻度直线固定 1 个原图像素宽，重叠处优先指针。原尺寸标签不增粗、不模糊、不预先缩放。训练中的补边/旋转空白区域使用 255 忽略标签；磁盘标签只有 0/1/2。

完全相同的源图文件或解码后完全相同的裁剪图归入同一组，禁止跨训练/验证/测试集。当前共有 5 对重复图片。相邻编号以及 pHash/灰度特征相似画面采用多次候选划分和等大小组交换，尽量分散到不同集合，同时保持各场景数量。不能保证每一对相邻图都分开，因训练集占比为 70%。具体分散统计见 manifest.json。

按本次指定规则，近似图片允许跨集合；评测结果反映这几个场景内的表现，不代表独立新场景的泛化能力。测试集不参与每轮模型选择。

重新整理其他版本的标注（已有输出不会覆盖）：

```bash
python3.12 -B -m src.data.prepare_dataset   --annotations-root ../segmentation_tool/dataset   --output data/pointer_scale_seg_v2 --seed 20261113
```

转换只依赖 NumPy、Pillow。训练服务器只需复制整理好的数据集，正式训练无需标注工具和原始 WSL 推理输出。

## 环境和预训练权重

使用现有 flowmeter_train:v1.1：PyTorch 2.7.1+cu128、Torchvision 0.22.1，训练代码另需 NumPy、Pillow。ONNX 导出另需 onnx；训练本身不需要 onnxruntime。不需要 Paddle、Ultralytics 或额外分割库。

默认 DeepLabV3+ 使用 ResNet18 编码器、输出步长 8、解码通道 128；支持改用 ResNet34 或输出步长 16。ASPP、低层特征融合和边界解码器共同输出三类全分辨率 logits。

默认复用之前下载的 ImageNet 编码器权重 resnet18-f37072fd.pth，放在工程 weights/ 下即可，无需重新下载。如果文件位于其他位置，可通过 WEIGHTS 指定。仅在缺少该文件时使用以下下载命令：

```bash
mkdir -p weights
curl -fL --retry 3 https://download.pytorch.org/models/resnet18-f37072fd.pth   -o weights/resnet18-f37072fd.pth
```

地址来自 [Torchvision 0.22.1 的官方 ResNet 权重定义](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/resnet.py)。默认训练只加载这个本地文件，不联网下载。若直接运行 Python，可显式传 --encoder-weights imagenet 自动下载，或用 --encoder-weights none 做流程检查；none 不推荐用于本次少量样本的正式训练。

## 宿主机启动训练容器

从工程根目录执行，源码、数据和权重只读挂载，输出可写。镜像只使用宿主机已有版本，不拉取、不构建、不安装软件：

```bash
# 单卡，默认 GPU 0、768 输入、全局 batch 4、最多 100 轮
bash src/training/train_docker.sh

# 两张 GPU，batch 是两张卡合计的值，每卡为 4
GPUS=0,1 bash src/training/train_docker.sh --batch 8

# 指定已有镜像和权重位置
IMAGE=flowmeter_train:v1.1 WEIGHTS=/path/resnet18-f37072fd.pth   bash src/training/train_docker.sh

# 显存不足时降低 batch 或输入尺寸
bash src/training/train_docker.sh --batch 2 --imgsz 512
```

DATASET、OUTPUT_ROOT 可覆盖数据和输出路径。如需改用 ResNet34，显式传入 --backbone resnet34，并将 WEIGHTS 指向对应的 [ResNet34 权重](https://download.pytorch.org/models/resnet34-b627a593.pth)。多卡的全局 batch 必须能被 GPU 数量整除。多进程通过 127.0.0.1 静态 rendezvous 通信，避免离线容器依赖主机名解析；默认端口 29500。

## 已进入训练容器时

将工程挂载到 /app，在 /app 执行：

```bash
bash src/training/train.sh
GPUS=0,1 bash src/training/train.sh --batch 8

# Python 入口，路径均可配置
python3.12 -B -m src.training.train   --dataset data/pointer_scale_seg   --encoder-weights weights/resnet18-f37072fd.pth   --imgsz 768 --batch 4 --epochs 100
```

默认 AdamW：编码器学习率 3e-5，解码器 3e-4；编码器 BN 使用已有统计。损失为加权交叉熵（背景/指针/刻度 0.2/1/2）加前景 Dice。可配置 --class-weights、--dice-weight、--lr 和 --encoder-lr。增强包括同步旋转、轻度缩放/平移和颜色扰动；标签使用最近邻插值，不人为增粗刻度。

## 输出与验证

每次训练自动创建 outputs/pointer_scale_年月日_时分秒_微秒/（北京时间）：

- best.pt：验证集指针和刻度前景 mIoU 最高的检查点。
- last.pt：最近一轮检查点；仅包含模型及训练配置，不包含恢复优化器所需状态。
- results.csv：每轮训练损失、前景 mIoU、两类 IoU/Dice 和刻度 1 像素容差 F1。
- best_metrics.json、config.json、dataset_summary.json：对应指标、参数与数据划分摘要。

早停仅查看验证集（默认耐心 20 轮）。训练代码不自动反复评测测试集。需要输出最终测试集指标时，在容器内运行：

```bash
bash src/training/validate.sh   --checkpoint outputs/你的训练目录/best.pt --split test

# CPU 评测
bash src/training/validate.sh   --checkpoint outputs/你的训练目录/best.pt --split test --device cpu --workers 0
```

严格 IoU/Dice 之外保留 1 像素容差 F1，用于观察细刻度的轻微定位偏差；两类分别统计，不用背景准确率代替分割质量。

## 导出 ONNX

在含 onnx 的训练容器内执行：

```bash
bash src/training/export_onnx.sh --checkpoint outputs/你的训练目录/best.pt
```

生成 best.onnx 和 best.json。ONNX 固定输入 [1,3,H,W]、输出 [1,3,H,W] logits，opset 默认 17。类别顺序为背景、指针、刻度；部署端按 RGB、等比缩放补边和 ImageNet 均值/标准差预处理，再对 logits 的类别维 argmax，移除补边并以最近邻恢复原尺寸。best.json 保存这套输入输出约定。

## 本地检查记录

已检查 124 张图的人工确认状态、12 张删除排除、类别掩膜、场景计数、划分互斥及重复组不跨集合；转换掩膜与标注工具栅格化结果逐像素相同。

已有训练镜像内已通过 CPU 前向/反向、真实数据加载、单进程一轮小样本训练、CPU 双进程 DDP、检查点重载、测试集评测和 ONNX 导出。另使用现有推理镜像通过 ONNX Runtime 数值一致性检查。上述是随机初始化小样本的流程检查，不是正式训练效果，也未验证本次完整配置在 GPU 上的显存占用。
