using System;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Diagnostics;

public static class b7ExampleLibrary {
    [DllImport("../cpp_full_wrapper/build/libinfer.so", 
    CallingConvention = CallingConvention.Cdecl)]
    public static extern int infer_wrapper(string hef_path, string images_path, string arch, float conf_thr,
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

class Program {
    public const int FLOAT = 4;
    public const int MAX_NUM_DETECTIONS = 10; // was: 10
    public const int DETECTION_SIZE = 6; 
    public const int BUFFER_SIZE = 6; // In case of synchronization issues (c++ produces more that xBUFFER_SIZE faster than c# consumes)- make the buffer bigger. // was: 6; frames count is 1000; FPS: 5.5037127
    public const int CONF_IDX = 4;
    public const float CONF_THR = 0.5F;
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
        float conf_thr = CONF_THR;

        string[] extensions = { ".jpeg", ".jpg", ".png" };
        Regex regex = new Regex(string.Join("|", extensions.Select(ext => $"^{Regex.Escape(ext)}$")));
        int framesCount = Directory.GetFiles(imagesPath).Count(file => regex.IsMatch(Path.GetExtension(file)));
        
        Console.WriteLine("frames count is " + framesCount);

        Thread infer_thread = new Thread(() =>
        {
            DateTime startTime = DateTime.Now;
            int infer_result = b7ExampleLibrary.infer_wrapper(hefPath, imagesPath, arch, conf_thr, detections, max_num_detections, frames_ready, buffer_size);
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
                // Console.WriteLine("frame " + frame_idx + ", class: " + CocoClasses.CocoEightyClasses.Map[detection.class_id] + ", confidence: " + detection.confidence);
            }
            frames_ready[buffer_idx] = -1; // indicates that we have finished processing frame idx_buffer, and detections[buffer_idxdetections_size_per_frame] can be reused.
        }
        infer_thread.Join(); // Wait for infer_thread to complete
        Console.WriteLine(" b7 :)");
    }
}