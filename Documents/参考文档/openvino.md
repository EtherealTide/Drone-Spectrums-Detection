---
comments: true
description: 了解如何将 YOLO11 模型导出为 OpenVINO 格式，以获得最高 3 倍 CPU 加速，并在 Intel GPU 与 NPU 上启用硬件加速。
keywords: YOLO11, OpenVINO, 模型导出, Intel, AI 推理, CPU 加速, GPU 加速, NPU, 深度学习
---

# Intel OpenVINO 导出

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ecosystem.avif" alt="OpenVINO 生态系统">

本指南介绍如何将 YOLO11 模型导出为 [OpenVINO](https://docs.openvino.ai/) 格式，从而获得最高 3 倍的 [CPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) 推理加速，并在 Intel [GPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) 与 [NPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) 硬件上获得加速。

OpenVINO，全称 Open Visual Inference & [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization toolkit，是一个面向 AI 推理模型的全方位优化与部署工具包。尽管名称强调视觉，OpenVINO 同样支持语言、音频、时间序列等多种任务。

## 使用示例

导出 YOLO11n 模型为 OpenVINO 格式并运行推理。

!!! example

Python
```python
from ultralytics import YOLO

# 加载 YOLO11n PyTorch 模型
model = YOLO("yolo11n.pt")

# 导出模型
model.export(format="openvino")  # 创建 'yolo11n_openvino_model/'

# 加载导出的 OpenVINO 模型
ov_model = YOLO("yolo11n_openvino_model/")

# 运行推理
results = ov_model("https://ultralytics.com/images/bus.jpg")

# 指定设备运行推理，可选 ["intel:gpu", "intel:npu", "intel:cpu"]
results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
```

CLI

```bash
# 将 YOLO11n PyTorch 模型导出为 OpenVINO 格式
yolo export model=yolo11n.pt format=openvino # 创建 'yolo11n_openvino_model/'

# 使用导出的模型运行推理
yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'

# 指定设备运行推理，可选 ["intel:gpu", "intel:npu", "intel:cpu"]
yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg' device="intel:gpu"
```

## 导出参数

| 参数        | 类型              | 默认值        | 说明                                                                                                                                                                                         |
| ----------- | ----------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`             | `'openvino'`  | 导出模型的目标格式，决定与不同部署环境的兼容性。                                                                                                                                             |
| `imgsz`     | `int` 或 `tuple`  | `640`         | 模型输入图像尺寸。可为整数（方形图像）或元组 `(height, width)`。                                                                                                                            |
| `half`      | `bool`            | `False`       | 启用 FP16（半精度）量化，减少模型大小并在支持的硬件上提升推理速度。                                                                                                                         |
| `int8`      | `bool`            | `False`       | 启用 INT8 量化，进一步压缩模型并提升推理速度，对精度影响极小，主要用于边缘设备。                                                                                                           |
| `dynamic`   | `bool`            | `False`       | 允许动态输入尺寸，方便处理不同图像尺寸。                                                                                                                                                     |
| `nms`       | `bool`            | `False`       | 添加非极大值抑制（NMS），用于检测后处理。                                                                                                                                                    |
| `batch`     | `int`             | `1`           | 指定导出模型在 `predict` 模式下的批量推理大小或可同时处理的最大图像数。                                                                                                                     |
| `data`      | `str`             | `'coco8.yaml'`| [数据集](https://docs.ultralytics.com/datasets/) 配置文件路径，量化时必需。                                                                                                                  |
| `fraction`  | `float`           | `1.0`         | INT8 量化校准所用数据集比例，可用更小子集以降低资源消耗。若未指定则默认使用全量数据集。                                                                                                      |

更多导出流程信息参见 [Ultraytics 导出文档](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/export.md)

!!! warning

OpenVINO™ 兼容绝大多数 Intel® 处理器。为获得最佳性能请遵循以下步骤：

1. **验证 OpenVINO™ 支持**  
    通过 [Intel 兼容性列表](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html)确认处理器是否在支持列表中。

2. **识别加速器**  
    参考 [Intel 硬件指南](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html)确定处理器是否集成 NPU 或 GPU。

3. **安装最新驱动**  
    若设备支持 NPU/GPU 但 OpenVINO™ 无法识别，请按 [驱动安装指南](https://medium.com/openvino-toolkit/how-to-run-openvino-on-a-linux-ai-pc-52083ce14a98) 更新驱动。

## OpenVINO 的优势

1. **性能**：充分利用 Intel CPU、集成/独立 GPU 与 FPGA，实现高性能推理。
2. **异构执行**：一次编写即可部署到任意受支持的 Intel 硬件（CPU、GPU、FPGA、VPU 等）。
3. **模型优化器**：可从 PyTorch、[TensorFlow](https://www.ultralytics.com/glossary/tensorflow)、TensorFlow Lite、Keras、ONNX、PaddlePaddle、Caffe 等框架导入、转换与优化模型。
4. **易用性**：提供 80+ [教程笔记本](https://github.com/openvinotoolkit/openvino_notebooks)，涵盖 YOLOv8 优化等主题。

## OpenVINO 导出结构

将模型导出为 OpenVINO 格式后会得到一个目录，包含：

1. **XML 文件**：描述网络拓扑。
2. **BIN 文件**：存储权重与偏置二进制数据。
3. **映射文件**：记录原模型输出张量与 OpenVINO 张量名称的映射。

这些文件可用于 OpenVINO 推理引擎。

## 部署中的 OpenVINO 导出使用

导出成功后可通过两种方式运行推理：

1. 使用 `ultralytics` 包，提供高层 API 并封装 OpenVINO Runtime。
2. 使用原生 `openvino` 包，实现更灵活的推理控制。

### 使用 Ultralytics 推理

`ultralytics` 包通过 `predict` 方法轻松调用导出的 OpenVINO 模型，并可指定设备（如 `intel:gpu`、`intel:npu`、`intel:cpu`）。

```python
from ultralytics import YOLO

# 加载导出的 OpenVINO 模型
ov_model = YOLO("yolo11n_openvino_model/")
# 指定设备运行推理
ov_model.predict(device="intel:gpu")
```

适合快速原型或无需自定义推理流程的部署。

### 使用 OpenVINO Runtime 推理

OpenVINO Runtime 提供统一 API，可在全部 Intel 硬件上推理，并支持硬件负载均衡、异步执行等高级特性。更多示例参见 [YOLO11 笔记本](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov11-optimization)。

部署应用一般步骤：

1. 通过 `core = Core()` 初始化 OpenVINO。
2. 使用 `core.read_model()` 加载模型。
3. 调用 `core.compile_model()` 编译模型。
4. 准备输入数据（图像、文本、音频等）。
5. 通过 `compiled_model(input_data)` 运行推理。

详见 [OpenVINO 文档](https://docs.openvino.ai/) 或 [API 教程](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb)。

## OpenVINO YOLO11 基准

Ultralytics 团队在多种模型格式与[精度](https://www.ultralytics.com/glossary/precision)下对 YOLO11 进行基准测试，涵盖不同 Intel 设备。

!!! note

    以下结果仅供参考，实际表现受硬件、软件环境及系统负载影响。
    
    测试使用 `openvino` Python 包版本 [2025.1.0](https://pypi.org/project/openvino/2025.1.0/)。

### Intel Core CPU

Intel® Core® 系列涵盖 i3（入门）、i5（中端）、i7（高端）、i9（旗舰），适配从日常办公到专业高负载的需求。每代均在性能、能效与特性上有所提升。

以下基准在第 12 代 Intel® Core® i9-12900KS CPU 上以 FP32 精度运行。

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-corei9.avif" alt="Core CPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式        | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | ----------- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch     | ✅   | 5.4       | 0.5071              | 21.00            |
| YOLO11n | TorchScript | ✅   | 10.5      | 0.5077              | 21.39            |
| YOLO11n | ONNX        | ✅   | 10.2      | 0.5077              | 15.55            |
| YOLO11n | OpenVINO    | ✅   | 10.4      | 0.5077              | 11.49            |
| YOLO11s | PyTorch     | ✅   | 18.4      | 0.5770              | 43.16            |
| YOLO11s | TorchScript | ✅   | 36.6      | 0.5781              | 50.06            |
| YOLO11s | ONNX        | ✅   | 36.3      | 0.5781              | 31.53            |
| YOLO11s | OpenVINO    | ✅   | 36.4      | 0.5781              | 30.82            |
| YOLO11m | PyTorch     | ✅   | 38.8      | 0.6257              | 110.60           |
| YOLO11m | TorchScript | ✅   | 77.3      | 0.6306              | 128.09           |
| YOLO11m | ONNX        | ✅   | 76.9      | 0.6306              | 76.06            |
| YOLO11m | OpenVINO    | ✅   | 77.1      | 0.6306              | 79.38            |
| YOLO11l | PyTorch     | ✅   | 49.0      | 0.6367              | 150.38           |
| YOLO11l | TorchScript | ✅   | 97.7      | 0.6408              | 172.57           |
| YOLO11l | ONNX        | ✅   | 97.0      | 0.6408              | 108.91           |
| YOLO11l | OpenVINO    | ✅   | 97.3      | 0.6408              | 102.30           |
| YOLO11x | PyTorch     | ✅   | 109.3     | 0.6989              | 272.72           |
| YOLO11x | TorchScript | ✅   | 218.1     | 0.6900              | 320.86           |
| YOLO11x | ONNX        | ✅   | 217.5     | 0.6900              | 196.20           |
| YOLO11x | OpenVINO    | ✅   | 217.8     | 0.6900              | 195.32           |

### Intel® Core™ Ultra

Intel® Core™ Ultra™ 系列以融合 CPU、GPU 与 NPU 的混合架构满足游戏、创作及 AI 工作负载。NPU 可提升本地 AI 推理效率，实现面向未来的智能计算。

以下基准在 Intel® Core™ Ultra™ 7 258V 与 7 265K 上，以 FP32 与 INT8 精度测试。

#### Intel® Core™ Ultra™ 7 258V

!!! tip "基准"

=== "集成 Intel® Arc™ GPU"

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-gpu.avif" alt="Intel Core Ultra GPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5052              | 32.27            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5068              | 11.84            |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.4969              | 11.24            |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5776              | 92.09            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5797              | 14.82            |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5751              | 12.88            |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6262              | 277.24           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6306              | 22.94            |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6126              | 17.85            |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6361              | 348.97           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6365              | 27.34            |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6242              | 20.83            |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6984              | 666.07           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6890              | 39.09            |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6856              | 30.60            |

=== "Intel® Lunar Lake CPU"

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-cpu.avif" alt="Intel Core Ultra CPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5052              | 32.27            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5077              | 32.55            |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.4980              | 22.98            |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5776              | 92.09            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5782              | 98.38            |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5745              | 52.84            |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6262              | 277.24           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6307              | 275.74           |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6172              | 132.63           |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6361              | 348.97           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6361              | 348.97           |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6240              | 171.36           |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6984              | 666.07           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6900              | 783.16           |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6890              | 346.82           |

=== "集成 Intel® AI Boost NPU"

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-npu.avif" alt="Intel Core Ultra NPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5052              | 32.27            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5085              | 8.33             |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.5019              | 8.91             |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5776              | 92.09            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5788              | 9.72             |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5710              | 10.58            |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6262              | 277.24           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6301              | 19.41            |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6124              | 18.26            |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6361              | 348.97           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6362              | 23.70            |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6240              | 21.40            |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6984              | 666.07           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6892              | 43.91            |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6890              | 34.04            |

#### Intel® Core™ Ultra™ 7 265K

!!! tip "基准"

=== "集成 Intel® Arc™ GPU"

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-gpu.avif" alt="Intel Core Ultra GPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5072              | 16.29            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5079              | 13.13            |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.4976              | 8.86             |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5771              | 39.61            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5808              | 18.26            |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5726              | 13.24            |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6258              | 100.65           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6310              | 43.50            |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6137              | 20.90            |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6367              | 131.37           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6371              | 54.52            |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6226              | 27.36            |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6990              | 212.45           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6884              | 112.76           |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6900              | 52.06            |

=== "Intel® Arrow Lake CPU"

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-cpu.avif" alt="Intel Core Ultra CPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5072              | 16.29            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5077              | 15.04            |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.4980              | 11.60            |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5771              | 39.61            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5782              | 33.45            |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5745              | 20.64            |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6258              | 100.65           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6307              | 81.15            |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6172              | 44.63            |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6367              | 131.37           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6409              | 103.77           |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6240              | 58.00            |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6990              | 212.45           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6900              | 208.37           |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6897              | 113.04           |

=== "集成 Intel® AI Boost NPU"

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-npu.avif" alt="Intel Core Ultra NPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5072              | 16.29            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5075              | 8.02             |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.3656              | 9.28             |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5771              | 39.61            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5801              | 13.12            |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5686              | 13.12            |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6258              | 100.65           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6310              | 29.88            |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6111              | 26.32            |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6367              | 131.37           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6356              | 37.08            |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6245              | 30.81            |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6990              | 212.45           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6894              | 68.48            |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6417              | 49.76            |

## Intel® Arc GPU

Intel® Arc™ 是面向高性能游戏、内容创作与 AI 工作负载的独立显卡系列，支持实时光线追踪、AI 增强图形、高分辨率游戏等特性，并提供 AV1 硬件编码与最新图形 API 支持。

以下基准在 Intel Arc A770 与 Arc B580 上以 FP32 与 INT8 精度运行。

### Intel Arc A770

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-a770-gpu.avif" alt="Intel Core Ultra CPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5072              | 16.29            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5073              | 6.98             |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.4978              | 7.24             |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5771              | 39.61            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5798              | 9.41             |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5751              | 8.72             |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6258              | 100.65           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6311              | 14.88            |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6126              | 11.97            |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6367              | 131.37           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6364              | 19.17            |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6241              | 15.75            |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6990              | 212.45           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6888              | 18.13            |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6930              | 18.91            |

### Intel Arc B580

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-b580-gpu.avif" alt="Intel Core Ultra CPU 基准">
</div>

??? abstract "详细基准结果"

| 模型    | 格式      | 精度 | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/图) |
| ------- | --------- | ---- | ---- | --------- | ------------------- | ---------------- |
| YOLO11n | PyTorch   | FP32 | ✅   | 5.4       | 0.5072              | 16.29            |
| YOLO11n | OpenVINO  | FP32 | ✅   | 10.4      | 0.5072              | 4.27             |
| YOLO11n | OpenVINO  | INT8 | ✅   | 3.3       | 0.4981              | 4.33             |
| YOLO11s | PyTorch   | FP32 | ✅   | 18.4      | 0.5771              | 39.61            |
| YOLO11s | OpenVINO  | FP32 | ✅   | 36.4      | 0.5789              | 5.04             |
| YOLO11s | OpenVINO  | INT8 | ✅   | 9.8       | 0.5746              | 4.97             |
| YOLO11m | PyTorch   | FP32 | ✅   | 38.8      | 0.6258              | 100.65           |
| YOLO11m | OpenVINO  | FP32 | ✅   | 77.1      | 0.6306              | 6.45             |
| YOLO11m | OpenVINO  | INT8 | ✅   | 20.2      | 0.6125              | 6.28             |
| YOLO11l | PyTorch   | FP32 | ✅   | 49.0      | 0.6367              | 131.37           |
| YOLO11l | OpenVINO  | FP32 | ✅   | 97.3      | 0.6360              | 8.23             |
| YOLO11l | OpenVINO  | INT8 | ✅   | 25.7      | 0.6236              | 8.49             |
| YOLO11x | PyTorch   | FP32 | ✅   | 109.3     | 0.6990              | 212.45           |
| YOLO11x | OpenVINO  | FP32 | ✅   | 217.8     | 0.6889              | 11.10            |
| YOLO11x | OpenVINO  | INT8 | ✅   | 55.9      | 0.6924              | 10.30            |

## 复现我们的结果

若要在所有导出[格式](../modes/export.md)上复现上述基准，可运行以下代码：

!!! example

    === "Python"
    
        ```python
        from ultralytics import YOLO
    
        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")
    
        # 在 COCO128 数据集上对所有导出格式进行速度和精度基准
        results = model.benchmark(data="coco128.yaml")
        ```
    
    === "CLI"
    
        ```bash
        # 在 COCO128 数据集上对所有导出格式进行速度和精度基准
        yolo benchmark model=yolo11n.pt data=coco128.yaml
        ```
    
    结果可能因硬件、软件配置及系统负载变化。要获得更可靠结果，请使用包含大量图像的数据集，例如 `data='coco.yaml'`（5000 张验证图像）。

## 结论

基准结果表明，将 YOLO11 导出为 OpenVINO 格式可显著提升推理速度，同时保持类似精度。OpenVINO 是部署深度学习模型的高效工具，使实际应用更易实现。

更多细节请参考 [OpenVINO 官方文档](https://docs.openvino.ai/)。

## FAQ

### 如何将 YOLO11 模型导出为 OpenVINO 格式？

导出可显著提升 CPU 推理速度，并启用 Intel GPU/NPU 加速，可使用 Python 或 CLI：

!!! example

    === "Python"
    
        ```python
        from ultralytics import YOLO
    
        model = YOLO("yolo11n.pt")
        model.export(format="openvino")  # 创建 'yolo11n_openvino_model/'
        ```
    
    === "CLI"
    
        ```bash
        yolo export model=yolo11n.pt format=openvino # 创建 'yolo11n_openvino_model/'
        ```

详见 [导出格式文档](../modes/export.md)。

### 为什么要在 YOLO11 模型上使用 OpenVINO？

优势包括：

1. **性能**：CPU 推理最高提升 3 倍，并可在 Intel GPU/NPU 上进一步加速。
2. **模型优化器**：支持从 PyTorch、TensorFlow、ONNX 等框架转换并优化模型。
3. **易用性**：提供 80+ 教程笔记本帮助快速上手。
4. **异构执行**：统一 API 部署到各类 Intel 硬件。

详情参见 [基准章节](#openvino-yolo11-基准)。

### 如何使用导出的 OpenVINO YOLO11 模型运行推理？

导出 YOLO11n 后，可使用 Python 或 CLI 推理：

!!! example

    === "Python"
    
        ```python
        from ultralytics import YOLO
    
        ov_model = YOLO("yolo11n_openvino_model/")
        results = ov_model("https://ultralytics.com/images/bus.jpg")
        ```
    
    === "CLI"
    
        ```bash
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

参阅 [predict 模式文档](../modes/predict.md)。

### 为什么选择 Ultralytics YOLO11 进行 OpenVINO 导出？

Ultralytics YOLO11 具备实时检测、精度高、速度快等特点，与 OpenVINO 结合可实现：

- 最高 3 倍 CPU 加速
- 无缝部署到 Intel GPU/NPU
- 在各导出格式间保持一致精度

更多分析见 [YOLO11 详细基准](#openvino-yolo11-基准)。

### 能否对不同格式（如 PyTorch、ONNX、OpenVINO）的 YOLO11 模型进行基准测试？

可以。使用以下代码即可对多种格式进行基准：

!!! example

Python

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.benchmark(data="coco8.yaml")
```

CLI

```bash
yolo benchmark model=yolo11n.pt data=coco8.yaml
```

更详细的结果请参见本页 [基准章节](#openvino-yolo11-基准)及 [导出格式文档](../modes/export.md)。