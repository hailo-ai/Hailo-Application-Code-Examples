using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Drawing;

public static class b7ExampleLibrary {
    [DllImport("/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/so/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string image_path, string arch,
    float[] detections, int max_num_detections);
}

public struct Detection
{
    public float ymin;
    public float xmin;
    public float ymax;
    public float xmax;
    public float confidence;
    public int class_id;

    public Detection(float[] arr, int offset)
    {
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
    public const float THR = 0.3F;

    static void Main() {
        ulong detections_size = MAX_NUM_DETECTIONS * SIZE_DETECTION; // * FLOAT
        float[] detections = new float[detections_size];
        int num_detections = MAX_NUM_DETECTIONS;

        // =========================================================
        int num_frames = 100;
        Stopwatch stopWatch = new Stopwatch();
        stopWatch.Start();
        for (int i = 0; i < num_frames; i++) {
           int infer_result = b7ExampleLibrary.infer_wrapper("yolov5m_wo_spp_60p_2.4.hef", "images/zidane_640.jpg", "yolov5", detections, num_detections);
        }
        stopWatch.Stop();
        TimeSpan ts = stopWatch.Elapsed;
        // string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
        //     ts.Hours, ts.Minutes, ts.Seconds,
        //     ts.Milliseconds / 10);
        Console.WriteLine("FPS " + num_frames/(ts.TotalSeconds));
        // =========================================================

        // int infer_result = b7ExampleLibrary.infer_wrapper("yolov5m_wo_spp_60p_2.4.hef", "images/zidane_640.jpg", "yolov5", detections, num_detections); 
        // // TODO: return num of actual detections. That's super important to not read 'junk values'.
        // if (infer_result != 0) {
        //     Console.WriteLine("Inference failed with error code: " + infer_result);
        //     return;
        // }

        // for (int idx_detection = 0; idx_detection < num_detections; idx_detection++) 
        // {
        //     if ( detections[ idx_detection*SIZE_DETECTION + CONF_IDX ] >= THR ) { // confidence >= thr // no need if adding only good detections in c++
        //         Detection detection = new Detection(detections, idx_detection*SIZE_DETECTION);
        //     } 
        // }

        // ==========================================================

        // float detections_0 = detections[0];
        // float detections_1 = detections[1];

        // for (ulong i = 0; i < detections_size; i += SIZE_DETECTION) // for (ulong i = 4; i < detections_size; i=i+6)
        // {
        //     float ymin = detections[i];
        //     float xmin = detections[i+1];
        //     float ymax = detections[i+2];
        //     float xmax = detections[i+3];
        //     float confidence = 
        //     detections[i+4] = detection.confidence;
        //     detections[i+5] = static_cast<float32_t>(detection.class_id);
        //     if (detections[i] >= 0.3) {
        //         Console.WriteLine(i + ": " + detections[i]);
        //     }
        // }

        // Console.WriteLine("values: " + detections_0 + ", " + detections_1 + ", infer result: " + infer_result);
        
    }
}