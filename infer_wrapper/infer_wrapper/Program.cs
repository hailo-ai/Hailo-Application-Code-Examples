using System;
using System.Runtime.InteropServices;

public static class b7ExampleLibrary {
    [DllImport("/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string images_path, 
    float[] detections, ulong detections_size);
}

class Program {
    public const int FLOAT = 4;
    public const int NUM_DETECTIONS = 20;
    public const int SIZE_DETECTION = 6;

    static void Main() {
        ulong detections_size = NUM_DETECTIONS * SIZE_DETECTION;
        float[] detections = new float[detections_size];

        int infer_result = b7ExampleLibrary.infer_wrapper("yolov5m_wo_spp_60p.hef", "images", detections, detections_size);

        float arr1_0 = arr1[0];
        float arr1_1 = arr1[1];

        for (ulong i = 4; i < n1; i=i+5)
        {
            if (arr1[i] >= 0.3) {
                Console.WriteLine(i + ": " + arr1[i]);
            }
        }

        Console.WriteLine("values: " + arr1_0 + ", " + arr1_1 + ", infer result: " + infer_result);
        
    }
}