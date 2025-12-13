# miso-onnx
Inference using ONNX for image classification

## Installation

conda create -n miso-onnx python=3.11

conda activate miso-onnx

pip install git+https://github.com/microfossil/particle-classification-onnx

## Command line interface (CLI)

Classify a folder of images using the `network_info.xml` in the `model_onnx` folder

```
miso-onnx classify --network-info path/to/network_info.xml --images path/to/images --output-csv path/to/output.csv --output-json path/to/output.json --device cpu
```

Use --device cuda for GPU inference, note:

```
Requires cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.
```

Full usage:

```
Usage: miso-onnx classify [OPTIONS]

  Classify images using an ONNX model.

  You can provide either:

  1. A network info XML file (--network-info) that contains all configuration,
  OR

  2. Individual parameters (--model and --labels)

  Examples:

      # Using network info XML:     python -m miso-onnx classify --network-
      info model/network_info.xml --images data/images/

      # Using individual parameters:     python -m miso-onnx classify --model
      model.onnx --labels "cat,dog,bird" --images data/images/

      # With output files:     python -m miso-onnx classify --network-info
      model/network_info.xml --images data/images/ \         --output-json
      results.json --output-csv predictions.csv

      # Using CUDA:     python -m miso-onnx classify --network-info
      model/network_info.xml --images data/images/ \         --device cuda
      --batch-size 64

Options:
  --network-info PATH             Path to network info XML file (contains all
                                  model configuration)
  --model PATH                    Path to ONNX model file (required if not
                                  using --network-info)
  --labels TEXT                   Comma-separated labels or path to labels
                                  file (required if not using --network-info)
  --images DIRECTORY              Path to folder containing images to classify
                                  [required]
  --batch-size INTEGER            Batch size for inference (default: 32)
  --num-workers INTEGER           Number of parallel workers for image loading
                                  (default: 4)
  --device [cpu|cuda]             Device to run inference on (default: cpu)
  --output-json PATH              Path to save results as JSON
  --output-csv PATH               Path to save predictions as CSV
  --show-progress / --no-progress
                                  Show progress bars during processing
                                  (default: show)
  --help                          Show this message and exit.
```