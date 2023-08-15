## espcn inference example
This example shows SR (Super-Resolution) inference example.  
It uses the model espcn_x4_540_960.hef
It takes image in 540x960 resolution and enlarged it to 4k resolution.

## Requirements
HailoRT  (tested on 4.13.0)   
loguru   (tested on 0.7.0)
## Usage
```
./espcn_infer.py --hef espcn_x4_540x960.hef --image images/ -o output_dir
```   
