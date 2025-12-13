# miso-onnx
Inference using ONNX for image classification

## Installation

conda create -n miso-onnx python=3.11

conda activate miso-onnx

pip install git+https://github.com/microfossil/particle-classification-onnx

## Command line interface (CLI)

Classify a folder of images using the network_info.xml

```
miso-onnx classify --network-info path/to/network_info.xml --images path/to/images --output-csv path/to/output.csv --output-json path/to/output.json --device cpu
```

Use --device cuda for GPU inference, note:

```
Requires cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.
```