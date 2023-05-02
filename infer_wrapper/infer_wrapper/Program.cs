using System;
using System.Runtime.InteropServices;

public static class b7ExampleLibrary {
    [DllImport("/local/users/batshevak/projects/b7_git/Hailo-Application-Code-Examples/infer_wrapper/infer_wrapper/libinfer.so")]
    public static extern int infer_wrapper(string hef_path, string images_path);
    // public static extern int infer_wrapper_test(int a);

    // [DllImport("/local/users/batshevak/projects/infer_wrapper_b7/libinfer.so")]
    // public static extern int add(int x, int y);
}

namespace HelloWorld
{
    class b7_Program {
        static void Main() {
            Console.WriteLine("Hello, world!");
            // int result = b7ExampleLibrary.add(3, 4);
            // Console.WriteLine(result); // output: 7
            Console.WriteLine("b7");
            int infer_result = b7ExampleLibrary.infer_wrapper("resnet_v1_18.hef", "bla");
            // int infer_result = b7ExampleLibrary.infer_wrapper_test(3);
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

