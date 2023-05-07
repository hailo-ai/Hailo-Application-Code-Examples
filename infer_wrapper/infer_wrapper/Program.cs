using System;
using System.Runtime.InteropServices;

public static class b7ExampleLibrary {
    [DllImport("/local/users/batshevak/projects/b7_git/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string images_path, 
    float[] arr1, ulong n1,
    float[] arr2, ulong n2,
    float[] arr3, ulong n3);
}

class Program {
    static void Main() {
        float[] arr1 = new float[5];
        ulong n1 = 5;
        float[] arr2 = new float[5];
        ulong n2 = 5;
        float[] arr3 = new float[5];
        ulong n3 = 5;

        int infer_result = b7ExampleLibrary.infer_wrapper("yolov5m_wo_spp_60p.hef", "images", arr1, n1, arr2, n2, arr3, n3);

        float arr1_0 =arr1[0];
        float arr1_1 = arr1[1];

        Console.WriteLine("values: " + arr1_0 + ", " + arr1_1 + ", infer result: " + infer_result);
    }
}