using System;
using System.Runtime.InteropServices;

public static class b7ExampleLibrary {
    [DllImport("/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/so/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string image_path, string arch,
    float[] detections, int max_num_detections);
}

class Program {
    public const int FLOAT = 4;
    public const int NUM_DETECTIONS = 20;
    public const int SIZE_DETECTION = 6;

    static void Main() {
        ulong detections_size = NUM_DETECTIONS * SIZE_DETECTION; // * FLOAT
        float[] detections = new float[detections_size];
        int num_detections = NUM_DETECTIONS;

        int infer_result = b7ExampleLibrary.infer_wrapper("yolov5m_wo_spp_60p.hef", "images/zidane_640.jpg", "yolov5", detections, num_detections); // TODO: return num of actual detections

        float detections_0 = detections[0];
        float detections_1 = detections[1];

        for (ulong i = 4; i < detections_size; i=i+6)
        {
            if (detections[i] >= 0.3) {
                Console.WriteLine(i + ": " + detections[i]);
            }
        }

        Console.WriteLine("values: " + detections_0 + ", " + detections_1 + ", infer result: " + infer_result);
        
    }
}