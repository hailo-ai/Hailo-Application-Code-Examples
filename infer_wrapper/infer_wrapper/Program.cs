using System;
using System.Runtime.InteropServices;

public static class b7ExampleLibrary {
    [DllImport("/home/batshevak/projects/new_jj/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string images_path, 
    float[] arr1, ulong n1,
    float[] arr2, ulong n2,
    float[] arr3, ulong n3);
}

class Program {
    public const int FEATURE_MAP_SIZE1 = 20;
    public const int FEATURE_MAP_SIZE2 = 40;
    public const int FEATURE_MAP_SIZE3 = 80;
    public const int FEATURE_MAP_CHANNELS = 85;
    public const int ANCHORS_NUM = 3;
    public const int FLOAT = 4;

    static void Main() {
        ulong n1 = FEATURE_MAP_SIZE1 * FEATURE_MAP_SIZE1 * FEATURE_MAP_CHANNELS * ANCHORS_NUM * FLOAT;
        float[] arr1 = new float[n1];
        ulong n2 = FEATURE_MAP_SIZE2 * FEATURE_MAP_SIZE2 * FEATURE_MAP_CHANNELS * ANCHORS_NUM * FLOAT;
        float[] arr2 = new float[n2];
        ulong n3 = FEATURE_MAP_SIZE3 * FEATURE_MAP_SIZE3 * FEATURE_MAP_CHANNELS * ANCHORS_NUM * FLOAT;
        float[] arr3 = new float[n3];

        int infer_result = b7ExampleLibrary.infer_wrapper("yolov5m_wo_spp_60p.hef", "images", arr1, n1, arr2, n2, arr3, n3);

        float arr1_0 =arr1[0];
        float arr1_1 = arr1[1];

        Console.WriteLine("values: " + arr1_0 + ", " + arr1_1 + ", infer result: " + infer_result);
    }
}