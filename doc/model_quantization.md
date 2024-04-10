# ONNX model quantization

## Profile onnx inference

The code is from `xtrt\tools\quant\profile_onnx.py`: Before optimizing, run this script first to determine whether it's an issue with the GPU or if the CPU is slowing things down. If the problem lies with the CPU, optimizing the GPU might not significantly improve the overall optimization results.

> **Note:** Before using Tensorboard to analyze json files, you need to install `torch-tb-profiler` using pip.

### Setup

#### Custom Operation Handler

A custom operation handler named `efficientNMS_forward` is defined and registered for the `EfficientNMS_TRT` operation. This handler simulates the operation's forward pass, producing placeholder outputs.

```py
def efficientNMS_forward(*args, **kwards):
    return \
        torch.zeros([1, 1], dtype=torch.int32).cuda(), \
        torch.zeros([1, 100, 4],dtype=torch.float32).cuda(), \
        torch.zeros([1, 100],dtype=torch.float32).cuda(), \
        torch.zeros([1, 100], dtype=torch.int32).cuda() 
register_operation_handler(efficientNMS_forward, 'EfficientNMS_TRT', platform=TargetPlatform.FP32)
```

#### Model Quantization

Adjust `do_quantize` to `True` as needed for quantization. It uses a custom data loader and specifies the target platform as TRT_INT8 for potential quantization to INT8 precision. The `quantize_onnx_model` function from PPQ is utilized for this purpose. 

#### Profiling with Torch Profiler

The torch-profiler is used to profile the model's performance across 16 batches of input data. The profiler is configured to record CPU and CUDA activities, including stack traces, for a detailed analysis of the model's execution.

### Execution Guide

1. **Environment Setup:** Confirm the installation of all necessary libraries, including PyTorch, PPQ, torch-profiler, and tqdm, within your Python environment.

2. **Model Preparation:** Ensure the ONNX model file (`yolov5s_trt8.onnx`) is located within the script's directory.

   ```py
   onnx_import_file='yolov5s_trt8.onnx',
   ```

3. **Code Adjustment:** For custom operators like `EfficientNMS_TRT` present in the ONNX model, define a corresponding inference function for accurate model inference.

4. **Script Execution:** Run the script in your Python environment to perform both profiling and, optionally, model quantization.

5. **Analysis with TensorBoard:** Leverage TensorBoard to visualize and analyze profiling results for in-depth performance insights.

   ```sh
   tensorboard --logdir=log
   ```

### Expected Outcome

- **Quantization:** If `do_quantize` is set to `True`, the script will quantize the model to the specified precision (INT8 in this case), which is beneficial for deploying the model on platforms where reduced precision computation can significantly accelerate inference.
- **Profiling:** The script generates detailed profiling logs that can be viewed using TensorBoard. These logs provide insights into the model's performance, including execution time and resource utilization, which are crucial for optimization and debugging.

## Quantization ONNX

The code is from `xtrt\tools\quant\quant_onnx.py`. This script is designed for quantizing an ONNX model using the PPQ framework, tailored specifically for enhancing the deployment efficiency of neural network models on hardware platforms that benefit from reduced precision computations, such as GPUs supporting INT8 operations. The script includes several key stages, from initial setup and custom operation handling to dataset preparation, model quantization, error analysis, and finally, exporting the quantized model. 

### Script Configuration

#### Initial Setup

Before running the script, you must adjust several configuration variables to match your specific requirements:

- `DATA_DIRECTORY`: Path to the COCO dataset.
- `ONNX_IMPORT_FILE`: Path to the pre-trained ONNX model.
- `ONNX_EXPORT_FILE`: Directory where the quantized model and configuration will be saved.
- `TARGET_PLATFORM` and `EXPORT_PLATFORM`: Define the target platform for quantization and the format for the exported model.
- `MODEL_TYPE`: Specifies the model framework (ONNX or CAFFE).
- `INPUT_LAYOUT`: The input data layout (CHW or HWC).
- `NETWORK_INPUTSHAPE`: The expected input shape of the network.
- `CALIBRATION_BATCHSIZE`: The batch size for the calibration dataset.
- `EXECUTING_DEVICE`: Specifies the device ('cuda' or 'cpu') for execution.

#### Quantization Setting (QS)

The `QuantizationSettingFactory.default_setting()` function generates a default quantization setting, which is then customized based on whether you wish to fine-tune the network. LSQ optimization can be enabled for training the network further.

#### Custom Operation Handler

For ONNX models containing custom operations like `EfficientNMS_TRT`, you need to define a forward function that simulates these operations and register it with PPQ. This ensures accurate quantization and inference.

#### Dataset Preparation

A subset of the COCO validation dataset is loaded and transformed using a combination of custom (`Letterbox`) and standard (`ToTensor`) transformations to match the network input requirements.

### Running the Script

#### Data Loader

A `DataLoader` is created for the transformed COCO dataset, which is then used for model calibration during quantization.

#### Model Quantization

The script quantizes the model by:

- Loading the ONNX model graph.
- Adjusting quantization settings based on the network's needs.
- Running the quantization process with the COCO dataset for calibration.
- Optionally, calculating and analyzing quantization error to ensure model accuracy.

#### Error Analysis

After quantization, the script can perform both graph-wise and layer-wise error analyses to identify potential accuracy issues caused by quantization noise. This step is crucial for maintaining model performance.

#### Model Export

Finally, the quantized model, along with its configuration, is exported to the specified directory. The export format is adjusted based on the target platform chosen.

### Outputs

Upon successful completion, the script generates:

- A quantized version of the ONNX model (`quantized.onnx`).
- A quantization configuration file (`quant_cfg.json`).

These files are saved in the directory specified by `ONNX_EXPORT_FILE`.