using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public class TensorWrapper
{
    [MarshalAs (UnmanagedType.ByValArray, SizeConst=5)]
    public float[] array_5 = new float[5];

    [MarshalAs (UnmanagedType.ByValArray, SizeConst=6)]
    public float[] array_6 = new float[6];

    [MarshalAs (UnmanagedType.ByValArray, SizeConst=7)]
    public float[] array_7 = new float[7];

    public TensorWrapper() {}

    public TensorWrapper(float[] arr5, float[] arr6, float[] arr7)
    {
        if (arr5.Length != 5 || arr6.Length != 6 || arr7.Length != 7)
            throw new ArgumentException("Input arrays must have the correct length.");
        array_5 = arr5;
        array_6 = arr6;
        array_7 = arr7;
    }
}

// class TensorWrapper {
//     [MarshalAs (UnmanagedType.ByValArray, SizeConst=5)]
//     public float[] array_5 = new float[5];

//     [MarshalAs (UnmanagedType.ByValArray, SizeConst=6)]
//     public float[] array_6 = new float[6];
    
//     [MarshalAs (UnmanagedType.ByValArray, SizeConst=7)]
//     public float[] array_7 = new float[7];
// }

public static class b7ExampleLibrary {
    [DllImport("/local/users/batshevak/projects/b7_git/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/libinfer2.so", 
    CallingConvention = CallingConvention.Cdecl)]
    // [MarshalAs (UnmanagedType.ByValArray, SizeConst=10)]
    // public int[]  data;
    public static extern int infer_wrapper(string hef_path, string images_path, [In, Out] ref TensorWrapper tensor); // [In, Out] TensorWrapper tensor
}

// /* PassByReferenceInOut */
//     [DllImport ("mylib")]
//     public static extern
//        void PassByReferenceInOut ([In, Out] ClassWrapper s);

namespace HelloWorld
{
    class b7_Program {
        static void Main() {
            Console.WriteLine("Hello, world!");
            // int result = b7ExampleLibrary.add(3, 4);
            // Console.WriteLine(result); // output: 7
            Console.WriteLine("b7");
            TensorWrapper tensor = new TensorWrapper();
            int infer_result = b7ExampleLibrary.infer_wrapper("resnet_v1_18.hef", "images", ref tensor);
            float arr5_0 = tensor.array_5[0];
            float arr5_1 = tensor.array_5[1];
            Console.WriteLine("values: ");
            Console.WriteLine(arr5_0);
            Console.WriteLine(arr5_1);
            Console.WriteLine(infer_result);
        }
    }
}


// public static class MyLibWrapper {
//     "/home/batshevak/projects/general/libb7.so"
//     [DllImport("libhailort.so")]
//     public static extern hailo_status hailo_scan_devices(hailo_scan_devices_params_t *params, hailo_device_id_t *device_ids,
//     size_t *device_ids_length);

//     // static void Main(string[] args) {
//     //     hailo_status result = hailo_scan_devices(42, 13);
//     //     Console.WriteLine("Result: {0}", result);
//     // }
//     private static extern void strncpy (StringBuilder dest,
//     string src, uint n);
    
//     private static void UseStrncpy ()
//     {
//         StringBuilder sb = new StringBuilder (256);
//         strncpy (sb, "this is the source string", sb.Capacity);
//         Console.WriteLine (sb.ToString());
//     }
// }

// namespace HelloWorld
// {
//     class Program {
//         static void Main() {
//             Console.WriteLine("Hello, world!");
//         }
//     }
// }

