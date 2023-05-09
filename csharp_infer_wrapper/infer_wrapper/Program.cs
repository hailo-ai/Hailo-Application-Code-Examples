using System;
using System.Runtime.InteropServices;
using System.Diagnostics;

public static class b7ExampleLibrary {
    [DllImport("../cpp_full_wrapper/build/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string image_path, string arch,
    float[] detections, int max_num_detections);
}

public struct Detection {
    public float ymin;
    public float xmin;
    public float ymax;
    public float xmax;
    public float confidence;
    public int class_id;

    public Detection(float[] arr, int offset) {
        ymin = arr[offset];
        xmin = arr[offset + 1];
        ymax = arr[offset + 2];
        xmax = arr[offset + 3];
        confidence = arr[offset + 4];
        class_id = (int)arr[offset + 5];
    }
}

class Program {
    public const int FLOAT = 4;
    public const int MAX_NUM_DETECTIONS = 20;
    public const int SIZE_DETECTION = 6;
    public const int CONF_IDX = 4;
    public const float THR = 0.5F; // TODO: parameter to infer_wrapper(). Currently const 0.5F.

    static void Main() {
        
        ulong detections_size = MAX_NUM_DETECTIONS * SIZE_DETECTION;
        float[] detections = new float[detections_size];
        int max_num_detections = MAX_NUM_DETECTIONS;
        string imagePath = "images/zidane_640.jpg";
        string hefPath = "yolov5m_wo_spp_60p.hef";
        string arch = "yolov5";

        int infer_result = b7ExampleLibrary.infer_wrapper(hefPath, imagePath, arch, detections, max_num_detections); 
        // TODO: return num of actual detections. That's important to not read 'junk values' (values should be 0 if not initiallized, but may depend on compiler.)
        if (infer_result != 0) {
            Console.WriteLine("Inference failed with error code: " + infer_result);
            return;
        }

        for (int idx_detection = 0; idx_detection < max_num_detections; idx_detection++) {
            if ( detections[ idx_detection*SIZE_DETECTION + CONF_IDX ] >= THR ) { // no need to check after fixing TODO above 
                Detection detection = new Detection(detections, idx_detection*SIZE_DETECTION);
                Console.WriteLine("class: " + detection.class_id + ", confidence: " + detection.confidence);
            } 
        }
    }
}