using System;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

public static class InferLibrary {
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
    public int classId;

    public Detection(float[] arr, long offset) {
        ymin = arr[offset];
        xmin = arr[offset + 1];
        ymax = arr[offset + 2];
        xmax = arr[offset + 3];
        confidence = arr[offset + 4];
        classId = (int)arr[offset + 5];
    }
}

class Program {
    public const int FLOAT = 4;
    public const int MAX_NUM_DETECTIONS = 10;
    public const int DETECTION_SIZE = 6; 
    public const int BUFFER_SIZE = 6; // In case of synchronization issues (c++ produces more that xBUFFER_SIZE faster than c# consumes)- make the buffer bigger.
    public const int CONF_IDX = 4;
    public const float CONF_THR = 0.5F;
    public const int MILISEC_TO_WAIT = 1; // 0.001 sec

    static unsafe void Main() {
        
        int maxNumDetections = MAX_NUM_DETECTIONS;
        long detectionsSizePerFrame = MAX_NUM_DETECTIONS * DETECTION_SIZE;
        float[] detections = new float[detectionsSizePerFrame * BUFFER_SIZE];

        int bufferSize = BUFFER_SIZE;
        int[] framesReady = new int[bufferSize];
        Array.Fill(framesReady, -1); // all frames weren't processed yet
    
        string imagesPath = "images/";
        string hefPath = "yolov5m_wo_spp_60p.hef";
        string arch = "yolov5";
        float confThr = CONF_THR;

        string[] extensions = { ".jpeg", ".jpg", ".png" };
        Regex regex = new Regex(string.Join("|", extensions.Select(ext => $"^{Regex.Escape(ext)}$")));
        int framesCount = Directory.GetFiles(imagesPath).Count(file => regex.IsMatch(Path.GetExtension(file)));
        
        Console.WriteLine("frames count is " + framesCount);

        Thread inferThread = new Thread(() =>
        {
            DateTime startTime = DateTime.Now;
            int inferResult = InferLibrary.infer_wrapper(hefPath, imagesPath, arch, confThr, detections, maxNumDetections, framesReady, bufferSize);
            DateTime endTime = DateTime.Now;
            if (inferResult != 0) {
                Console.WriteLine("Inference failed with error code: " + inferResult);
                return;
            }
            TimeSpan elapsedTime = endTime - startTime;
            long elapsedMilliseconds = (long)elapsedTime.TotalMilliseconds;
            Console.WriteLine("FPS c# " + (framesCount*1000)/elapsedMilliseconds); // 1000: millisecs to secs
        });
        inferThread.Start();

        for (int frameIdx = 0; frameIdx < framesCount; frameIdx++) {
            int bufferIdx = frameIdx % bufferSize;
            while (framesReady[bufferIdx] == -1) {
                Thread.Sleep(MILISEC_TO_WAIT);
            }
            int numDetectionsFound = framesReady[bufferIdx];
            for (int idxDetection = 0; idxDetection < numDetectionsFound; idxDetection++) {
                Detection detection = new Detection(detections, bufferIdx*detectionsSizePerFrame + idxDetection*DETECTION_SIZE);
                Console.WriteLine("frame " + frameIdx + ", class: " + CocoClasses.CocoEightyClasses.Map[detection.classId] + ", confidence: " + detection.confidence);
            }
            framesReady[bufferIdx] = -1; // indicates that we have finished processing the frame frameIdx, and detections[bufferIdx] can be reused.
        }
        inferThread.Join(); // Wait for inferThread to complete
    }
}