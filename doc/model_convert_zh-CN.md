# MMYOLO 模型 ONNX 转换

> 本转化文档修改自[mmyolo](https://github.com/open-mmlab/mmyolo)的[`model_convert.md`](https://github.com/open-mmlab/mmyolo/blob/main/projects/easydeploy/docs/model_convert.md)文档。如果你想要使用下面的代码或者脚本进行导出你需要首先安装mmyolo才能进行，并且处在mmyolo的项目目录中。
>

## 1. 导出后端支持的 ONNX

### 环境依赖

- [onnx](https://github.com/onnx/onnx)

  ```shell
  pip install onnx
  ```

  [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) (可选，用于简化模型)

  ```shell
  pip install onnx-simplifier
  ```

> 请确保您在 `MMYOLO` 根目录下运行相关脚本，避免无法找到相关依赖包。

### 使用方法

[模型导出脚本](./projects/easydeploy/tools/export_onnx.py)用于将 `MMYOLO` 模型转换为 `onnx` 。

### 参数介绍

- `config` : 构建模型使用的配置文件，如 [`yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py`](./configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py) 。
- `checkpoint` : 训练得到的权重文件，如 `yolov5s.pth` 。
- `--work-dir` : 转换后的模型保存路径。
- `--img-size`: 转换模型时输入的尺寸，如 `640 640`。
- `--batch-size`: 转换后的模型输入 `batch size` 。
- `--device`: 转换模型使用的设备，默认为 `cuda:0`。
- `--simplify`: 是否简化导出的 `onnx` 模型，需要安装 [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)，默认关闭。
- `--opset`: 指定导出 `onnx` 的 `opset`，默认为 `11` 。
- `--backend`: 指定导出 `onnx` 用于的后端名称，`ONNXRuntime`: `onnxruntime`, `TensorRT8`: `tensorrt8`, `TensorRT7`: `tensorrt7`，默认为`onnxruntime`即 `ONNXRuntime`。
- `--pre-topk`: 指定导出 `onnx` 的后处理筛选候选框个数阈值，默认为 `1000`。
- `--keep-topk`: 指定导出 `onnx` 的非极大值抑制输出的候选框个数阈值，默认为 `100`。
- `--iou-threshold`: 非极大值抑制中过滤重复候选框的 `iou` 阈值，默认为 `0.65`。
- `--score-threshold`: 非极大值抑制中过滤候选框得分的阈值，默认为 `0.25`。
- `--model-only`: 指定仅导出模型 backbone + neck, 不包含后处理，默认关闭。

例子: 导出使用了`Efficient_NMS`的yolov5s onnx模型：

```shell
python ./projects/easydeploy/tools/export_onnx.py \
	configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py \
	yolov5s.pth \
	--work-dir work_dir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend tensorrt8 \ 
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```

然后利用后端支持的工具如 `TensorRT` 读取 `onnx` 再次转换为后端支持的模型格式如 `.engine/.plan` 等。

> `MMYOLO` 目前支持 `TensorRT8`, `TensorRT7`, `ONNXRuntime` 后端的端到端模型转换，**目前仅支持静态 shape 模型的导出和转换，动态 batch 或动态长宽的模型端到端转换会在未来继续支持。**

端到端转换得到的 `onnx` 模型输入输出如图：

![image-20240123172324590](../assets/image-20240123172324590.png)


输入名: `images`, 尺寸 640x640

输出名: `num_dets`, 尺寸 1x1，表示检测目标数量。

输出名: `boxes`, 尺寸 1x100x4，表示检测框的坐标，格式为 `x1y1x2y1`。

输出名: `scores`, 尺寸 1x100，表示检测框的分数。

输出名: `labels`, 尺寸 1x100，表示检测框的类别 id。

可以利用 `num_dets` 中的个数对 `boxes`, `scores`, `labels` 进行截断，从 100 个检测结果中抽取前 `num_dets` 个目标作为最终检测结果。

![image-20240123172147442](../assets/image-20240123172147442.png)

## 2. 仅导出模型 Backbone + Neck

当您需要部署在非 `TensorRT`, `ONNXRuntime` 等支持端到端部署的平台时，您可以考虑使用`--model-only` 参数并且不要传递 `--backend` 参数，您将会导出仅包含 `Backbone` + `neck` 的模型。

转换得到的 `onnx` 模型输入输出如图：

![image-20240123172455537](../assets/image-20240123172455537.png)


这种导出方式获取的 `ONNX` 模型具有如下优点:

- 算子简单，一般而言只包含 `Conv`，激活函数等简单算子，几乎不存在无法正确导出的情况，**对于嵌入式部署更加友好**。
- 方便不同算法之间对比速度性能，由于不同的算法后处理不同，仅对比 `backbone` + `Neck` 的速度更加公平。

也有如下缺点:

- 后处理逻辑需要单独完成，会有额外的 `decode` + `nms` 的操作需要实现。
- 与 `TensorRT` 相比，由于 `TensorRT` 可以利用多核优势并行进行后处理，可能使用 `--model-only` 方式导出的模型，使用单线处理的性能会差很多，但是我们可以写一些TensorRT Plugin来实现加速，实现比使用Efficient_NMS更快的速度。如使用`YoloLayer_TRT`参考：[Deepstream-YOLO](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/nvdsinfer_custom_impl_Yolo)

![image-20240123172739280](../assets/image-20240123172739280.png)

### 使用方法

```shell
python ./projects/easydeploy/tools/export_onnx.py \
	configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py \
	yolov5s.pth \
	--work-dir work_dir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--model-only
```



## 3. ONNX 推理测试

### 3.1 使用 `model-only` 导出的 ONNX 进行推理

[模型推理脚本](./projects/easydeploy/examples/main_onnxruntime.py)用于推理导出的 `ONNX` 模型，需要安装基础依赖环境:

[`onnxruntime`](https://github.com/microsoft/onnxruntime) 和 [`opencv-python`](https://github.com/opencv/opencv-python)

```shell
pip install onnxruntime
pip install opencv-python==4.7.0.72 # 建议使用最新的 opencv
```

#### 参数介绍

- `img` : 待检测的图片路径或图片文件夹路径。
- `onnx` : 导出的 `model-only` ONNX 模型。
- `--type` : 模型名称，目前支持 `yolov5`, `yolox`, `yolov6`, `ppyoloe`, `ppyoloep`, `yolov7`, `rtmdet`, `yolov8`。
- `--img-size`: 转换模型时输入的尺寸，如 `640 640`。
- `--out-dir`: 保存检测结果的路径 。
- `--show`: 是否可视化检测结果。
- `--score-thr`: 模型检测后处理的置信度分数 。
- `--iou-thr`: 模型检测后处理的 IOU 分数 。

#### 使用方法

```shell
cd ./projects/easydeploy/examples
python main_onnxruntime.py \
	"image_path_to_detect" \
	yolov5_s_model-only.onnx \
	--out-dir work_dir \
    --img-size 640 640 \
    --show \
    --score-thr 0.3 \
    --iou-thr 0.7
```

> ***注意！！！***
>
> 当您使用自定义数据集训练得到的模型时，请修改 [`config.py`](./projects/easydeploy/examples/config.py) 中 `CLASS_NAMES` 和 `CLASS_COLORS`，如果是 `yolov5` 或者 `yolov7` 基于 `anchor` 的模型请同时修改 `YOLOv5_ANCHORS` 和 `YOLOv7_ANCHORS`。
>
> [`numpy_coder.py`](./projects/easydeploy/examples/numpy_coder.py) 是目前所有算法仅使用 `numpy` 实现的 `decoder`，如果您对性能有较高的要求，可以参照相关代码改写为 `c/c++`。

### 3.2 使用 `backend tensorrt8 ` 导出的 ONNX 进行推理

将导出的onnx转化为TensorRT 的engine，然后推理，详细过程见应用代码分析部分。

<img src="../assets/image-20240123171754751.png" alt="image-20240123171754751" style="zoom:67%;" />

