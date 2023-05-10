
### Full Yolov5 Inference on hailo-8 with c# wrapper

## Usage   
```
# build so
cd cpp_full_wrapper 
./build.sh

cd ../infer_wrapper
# get hef file
./get_hef.sh
export LD_LIBRARY_PATH=`pwd`

# build & run c#
dotnet build -c Release
dotnet run

```
## Prerequirements   
1. OpenCV 4.2.X   
2. CMake >= 3.0.0   
3. HailoRT   

## More Info   
- Run full yolov5 inference on hailo-8 including post processing (nms)   
- libinfer.so get path to one image (jpg / png / jpeg), and return the detections found   
- Currently supports only one image   
- FPS not optimized, supposed for initial demonstration only.   
