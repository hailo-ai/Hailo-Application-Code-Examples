using System;
using System.Runtime.InteropServices;
using System.Diagnostics;

public static class b7ExampleLibrary {
    [DllImport("../cpp_full_wrapper/build/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string images_path, string arch,
    float[] detections, int max_num_detections, int[] frames_ready, int buffer_size);
}

public struct Detection {
    public float ymin;
    public float xmin;
    public float ymax;
    public float xmax;
    public float confidence;
    public int class_id;

    public Detection(float[] arr, long offset) {
        ymin = arr[offset];
        xmin = arr[offset + 1];
        ymax = arr[offset + 2];
        xmax = arr[offset + 3];
        confidence = arr[offset + 4];
        class_id = (int)arr[offset + 5];
    }
}

public static class CocoNinetyClasses 
{
    public static Dictionary<int, string> Map { get; } = new Dictionary<int, string>()
    {
        { 0, "unlabeled" },
        { 1, "person" },
        { 10, "traffic light" },
        { 11, "fire hydrant" },
        { 13, "stop sign" },
        { 14, "parking meter" },
        { 15, "bench" },
        { 16, "bird" },
        { 17, "cat" },
        { 18, "dog" },
        { 19, "horse" },
        { 20, "sheep" },
        { 2, "bicycle" },
        { 21, "cow" },
        { 22, "elephant" },
        { 23, "bear" },
        { 24, "zebra" },
        { 25, "giraffe" },
        { 27, "backpack" },
        { 28, "umbrella" },
        { 31, "handbag" },
        { 32, "tie" },
        { 33, "suitcase" },
        { 3, "car" },
        { 34, "frisbee" },
        { 35, "skis" },
        { 36, "snowboard" },
        { 37, "sports ball" },
        { 38, "kite" },
        { 39, "baseball bat" },
        { 40, "baseball glove" },
        { 41, "skateboard" },
        { 42, "surfboard" },
        { 43, "tennis racket" },
        { 4, "motorcycle" },
        { 44, "bottle" },
        { 46, "wine glass" },
        { 47, "cup" },
        { 48, "fork" },
        { 49, "knife" },
        { 50, "spoon" },
        { 51, "bowl" },
        { 52, "banana" },
        { 53, "apple" },
        { 54, "sandwich" },
        { 5, "airplane" },
        { 55, "orange" },
        { 56, "broccoli" },
        { 57, "carrot" },
        { 58, "hot dog" },
        { 59, "pizza" },
        { 60, "donut" },
        { 61, "cake" },
        { 62, "chair" },
        { 63, "couch" },
        { 64, "potted plant" },
        { 6, "bus" },
        { 65, "bed" },
        { 67, "dining table" },
        { 70, "toilet" },
        { 72, "tv" },
        { 73, "laptop" },
        { 74, "mouse" },
        { 75, "remote" },
        { 76, "keyboard" },
        { 77, "cellphone" },
        { 78, "microwave" },
        { 7, "train" },
        { 79, "oven" },
        { 80, "toaster" },
        { 81, "sink" },
        { 82, "refrigerator" },
        { 84, "book" }, 
        { 85, "clock" },
        { 86, "vase" },
        { 87, "scissors" },
        { 88, "teddy bear" },
        { 89, "hair drier" },
        { 8, "truck" },
        { 90, "tooth brush" },
        { 9, "boat" }
    };
}

public static class CocoEightyClasses 
{
    public static Dictionary<int, string> Map { get; } = new Dictionary<int, string>()
    {
        { 0, "unlabeled" },
        { 1, "person" },
        { 2, "bicycle" },
        { 3, "car" },
        { 4, "motorcycle" },
        { 5, "airplane" },
        { 6, "bus" },
        { 7, "train" },
        { 8, "truck" },
        { 9, "boat" },
        { 10, "traffic light" },
        { 11, "fire hydrant" },
        { 12, "stop sign" },
        { 13, "parking meter" },
        { 14, "bench" },
        { 15, "bird" },
        { 16, "cat" },
        { 17, "dog" },
        { 18, "horse" },
        { 19, "sheep" },
        { 20, "cow" },
        { 21, "elephant" },
        { 22, "bear" },
        { 23, "zebra" },
        { 24, "giraffe" },
        { 25, "backpack" },
        { 26, "umbrella" },
        { 27, "handbag" },
        { 28, "tie" },
        { 29, "suitcase" },
        { 30, "frisbee" },
        { 31, "skis" },
        { 32, "snowboard" },
        { 33, "sports ball" },
        { 34, "kite" },
        { 35, "baseball bat" },
        { 36, "baseball glove" },
        { 37, "skateboard" },
        { 38, "surfboard" },
        { 39, "tennis racket" },
        { 40, "bottle" },
        { 41, "wine glass" },
        { 42, "cup" },
        { 43, "fork" },
        { 44, "knife" },
        { 45, "spoon" },
        { 46, "bowl" },
        { 47, "banana" },
        { 48, "apple" },
        { 49, "sandwich" },
        { 50, "orange" },
        { 51, "broccoli" },
        { 52, "carrot" },
        { 53, "hot dog" },
        { 54, "pizza" },
        { 55, "donut" },
        { 56, "cake" },
        { 57, "chair" },
        { 58, "couch" },
        { 59, "potted plant" },
        { 60, "bed" },
        { 61, "dining table" },
        { 62, "toilet" },
        { 63, "tv" },
        { 64, "laptop" },
        { 65, "mouse" },
        { 66, "remote" },
        { 67, "keyboard" },
        { 68, "cell phone" },
        { 69, "microwave" },
        { 70, "oven" },
        { 71, "toaster" },
        { 72, "sink" },
        { 73, "refrigerator" },
        { 74, "book" },
        { 75, "clock" },
        { 76, "vase" },
        { 77, "scissors" },
        { 78, "teddy bear" },
        { 79, "hair drier" },
        { 80, "toothbrush" }
    };
}

class Program {
    public const int FLOAT = 4;
    public const int MAX_NUM_DETECTIONS = 10; // was: 10
    public const int DETECTION_SIZE = 6; 
    public const int BUFFER_SIZE = 6; // In case of synchronization issues (c++ produces more that xBUFFER_SIZE faster than c# consumes)- make the buffer bigger. // was: 6; frames count is 1000; FPS: 5.5037127
    public const int CONF_IDX = 4;
    public const float THR = 0.5F; // TODO: parameter to infer_wrapper(). Currently const 0.5F.
    public const int MILISEC_TO_WAIT = 1; // 0.001 sec

    static unsafe void Main() {
        
        int max_num_detections = MAX_NUM_DETECTIONS;
        long detections_size_per_frame = MAX_NUM_DETECTIONS * DETECTION_SIZE;
        float[] detections = new float[detections_size_per_frame * BUFFER_SIZE];

        int buffer_size = BUFFER_SIZE;
        int[] frames_ready = new int[buffer_size];
        Array.Fill(frames_ready, -1); // all frames weren't processed yet
    
        string imagesPath = "images/rand_coco_1000/images/calib_set/"; // was images/random_coco/
        string hefPath = "yolov5m_wo_spp_60p.hef";
        string arch = "yolov5";

        int framesCount = Directory.GetFiles(imagesPath, "*.jpg", SearchOption.TopDirectoryOnly).Length;
        Console.WriteLine("frames count is " + framesCount);

        Thread infer_thread = new Thread(() =>
        {
            DateTime startTime = DateTime.Now;
            int infer_result = b7ExampleLibrary.infer_wrapper(hefPath, imagesPath, arch, detections, max_num_detections, frames_ready, buffer_size);
            DateTime endTime = DateTime.Now;
            if (infer_result != 0) {
                Console.WriteLine("Inference failed with error code: " + infer_result);
                return;
            }
            TimeSpan elapsedTime = endTime - startTime;
            long elapsedMilliseconds = (long)elapsedTime.TotalMilliseconds;
            Console.WriteLine("FPS c# " + (framesCount*1000)/elapsedMilliseconds); // 1000: millisecs to secs
        });
        infer_thread.Start();

        for (int frame_idx = 0; frame_idx < framesCount; frame_idx++) {
            int buffer_idx = frame_idx % buffer_size;
            while (frames_ready[buffer_idx] == -1) {
                Thread.Sleep(MILISEC_TO_WAIT);
            }
            int num_detections_found = frames_ready[buffer_idx];
            for (int idx_detection = 0; idx_detection < num_detections_found; idx_detection++) {
                Detection detection = new Detection(detections, buffer_idx*detections_size_per_frame + idx_detection*DETECTION_SIZE);
                Console.WriteLine("frame " + frame_idx + ", class: " + CocoEightyClasses.Map[detection.class_id] + ", confidence: " + detection.confidence);
            }
            frames_ready[buffer_idx] = -1; // indicates that we have finished processing frame idx_buffer, and detections[buffer_idxdetections_size_per_frame] can be reused.
        }
        infer_thread.Join(); // Wait for infer_thread to complete
        Console.WriteLine(" b7 :)");
    }
}