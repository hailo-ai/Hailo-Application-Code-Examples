# Hailo Optimization diagnostic tool

First Aid Kit for common quantization and optimization issues
The tool was tested on `Hailo Dataflow Compiler v3.25.0`

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
                        Target HW arch {hailo8, hailo8r, hailo8l, hailo15h}
  --log-path LOG_PATH   Default path: diagnostic_tool.log

Advanced:
  Advanced diagnostic tool features

  --order INSPECTOR [INSPECTOR ...]
                        Choose which inspectors to run and set a custom order {NormInspector, CompressionInspector, ImageInspector, BatchNormInspector, ConcatenatedOutputsInspector, ClippingInspector,
                        HighPrecisionInspector}
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

The `--order` flag can be used to select custom inspector modules.

#### Add custom inspector

A custom inspector can be added by adding a new python module in inspectors/ and subclassing the BaseInspector module.

## Modules Explanation

The diagnostic tool is composed of different inspectors, each module assists in identifying common issues within the model.

### NormInspector

- Search for normalization layers at the start of the model.
- Check the input, and post normalization data distributions.

### CompressionInspector

- Inform the user of any compression.
- If no explicit compression level was detect, warns of implicit compression.

### ConcatenatedOutputsInspector

- Having some layers concatenated prior the output layer, forces an expanded dynamic range for the concatenated layers, which might reduce the accuracy. Selecting the pre-concat layers can improve the quantization results.

### HighPrecisionInspector

- Simple heurstic for assisting with 16 bit layers configuration

[//]: <> (Should use the advanced LAT results, or maybe 2 different inspectors?)

### ClippingInspector

- The clipping inspectors informs the user if more than 5% of the range (warning if more than 20%) is occupied by less than 3 or less values. It is highly recommended to use the same calibration data used of quantization / optimization. 

### BatchNormInspector

- The batch norm inspector checks if the layers of the model has fused batch norm layers.
- In case the model is exported from torch, the batch norm layers might be folded pre-export (and won't appear in the hn). If dataset sample is provided, pre-act distriubtion will be collected and checked for likelyhood of folded batch-norm in the layers.

### ImageInspector

- Saves few image samples as RGB and as BGR (standard.jpg, inverse.jpg). if the inverse file seems more natural, it means the data has been saved as BGR format. If none of the samples look natural, the images might be stored as YUV, YUY2, or other formats.

## Code

The tool is composed of "Inspector" modules, each Inspector diagnoses different common issue. If you'd like to read more about the inspectors, read [here](inspectors/README.md)
