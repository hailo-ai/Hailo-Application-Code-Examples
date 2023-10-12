# Hailo Optimization diagnostic tool

First Aid Kit for common quantization and optimization issues

## Usage

[//]: <> (markdown lacks the feature of embedding text from other file, maybe use rst files instead?)

```
usage: main.py [-h] [-d DATASET] [-a HW_ARCH] [--log-path LOG_PATH] [--order INSPECTOR [INSPECTOR ...]] har

positional arguments:
  har                   Path to HAR file

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Calibration Dataset, npy / npz file formats
  -a HW_ARCH, --hw_arch HW_ARCH
                        Target HW arch {hailo8, hailo8r, hailo8l, hailo15h, hailo15m}
  --log-path LOG_PATH   Default path: diagnostic_tool.log

Advanced:
  Advanced diagnostic tool features

  --order INSPECTOR [INSPECTOR ...]
                        Choose which inspectors to run and set a custom order {normalization, compression, output_concat, high_precision, clipping}
```

Example:

```
./main.py -d calibset.npy resnet_v1_18.har
```

### Expected output

The output of the diagnostic tool should look like the following output:

```
[info] Running inspector module: NormInspector
[info] Running inspector module: CompressionInspector
[info] Running inspector module: ConcatenatedOutputsInspector
[info] Running inspector module: HighPrecisionInspector
```

There might be additional info or warning messages between the inspector modules, for example:

```
[info] Running inspector module: NormInspector
[warning] Input layer resnet_v1_18/input_layer1 doesn't have normalization. Was the data normalized manually?
[error] Unexpected data distribution at resnet_v1_18/input_layer1. (expected normalized data) mean: [120.55961  114.191986  94.09033 ], std: [59.070805 56.93998  57.789307]
[error] Mean or std are above threshold. Data might not be noramlized
```

Warning messages doesn't indicate there're necesarily issues with the model, but it warns about non-standard behavior. In the current example, if the data is being normalized outside of the Hailo pipeline, the behavior should be completely fine, otherwise you might want to add normalization layer to your model.

The error messages usually indicates that the behavior is most likely problematic. In this example, the model model didn't have normalization and the input data wasn't normalized.

### Advanced usage

## Modules Explanation

## Code

The tool is composed of "Inspector" modules, each Inspector diagnoses different common issue. If you'd like to read more about the inspectors, read [here](inspectors/README.md)