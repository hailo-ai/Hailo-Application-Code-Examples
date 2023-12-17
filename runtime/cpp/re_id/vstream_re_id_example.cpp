/**
 * Copyright 2020 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/
/**
 * @ file vstreams_example
 * This example demonstrates using virtual streams over c++
 **/

#include "hailo/hailort.hpp"
#include "common/yolo_output.hpp"
#include "common/yolo_postprocess.hpp"
#include "common/re_id_overlay.hpp"
#include "common/hailo_common.hpp"
#include "common/hailo_objects.hpp"

#include <cxxabi.h>
#include <iostream>
#include <chrono>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp> 

#include <vector>
#include <string>
#include <fstream>
#include <array>
#include <typeinfo>
#include <iomanip>
#include <thread>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xsort.hpp"
#include "common/math.hpp"
#include "common/tensors.hpp"

constexpr bool QUANTIZED = false;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
constexpr int CV_32F_TYPE = CV_32FC3; //CV_8UC3; //CV_32FC3;
constexpr int CV_8U_TYPE = CV_8UC3;
std::mutex m;
std::mutex m2;
std::mutex m3;

#define CONFIG_FILE ("yolov5.json")
//#define CONFIG_FILE2 ("yolov5personFace.json")
#define MAX_BOXES 50
#define TEXT_FONT_FACTOR (0.12f)
#define DEFAULT_DETECTION_COLOR (cv::Scalar(255, 255, 255))

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */


using namespace hailort;
using hailort::Device;
using hailort::Hef;
using hailort::Expected;
using hailort::make_unexpected;
using hailort::ConfiguredNetworkGroup;
using hailort::VStreamsBuilder;
using hailort::InputVStream;
using hailort::OutputVStream;
using hailort::MemoryView;

#define PERSON_FACE_INFER   1
#define OSNET_IFER          2
#define PERSON_DETECTION    1

int num_of_detections_per_frame = 0;
int num_of_frames_to_process = 300;
int unique_id_counter = 1;

// the database to save the Re-Id vectors
std::vector<std::vector<HailoMatrixPtr>> reIdDatabase;
std::map<int, int> tracking_id_to_global_id;
std::vector<HailoDetectionPtr> personDetections;

float similarity_thr = 0.15;
uint queue_size = 100;


/**
 * @brief Print inference statistics to output terminal
 *
 * @param num_of_frames number of frames to be printed
 * @param elapsed_time time of inference to be printed
 * @return none
 */
void print_inference_statistics(std::size_t num_of_frames, double elapsed_time) {
    std::cout << BOLDGREEN << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Total Time:               " << elapsed_time << " sec" << std::endl;
    std::cout << "-I- Average FPS:              " << (double)num_of_frames / elapsed_time << std::endl;
    std::cout << "-I- Total Latency:            " << 1.0 / ((double)num_of_frames / elapsed_time)*1000 << " ms" << std::endl;
    std::cout << "-I-----------------------------------------------\n" << std::endl << RESET;
}

/**
 * @brief Post process after calling yolov5 person face detection.
 *
 * @param roi a pointer to the region of interest
 * @param image a matrix containing the image to process
 * @return hailo_status
 */
hailo_status post_process(HailoROIPtr& roi, cv::Mat &image) {
    YoloParams* init_params;
    init_params = init(CONFIG_FILE);
 
    yolov5_personface(roi, init_params);

    // call the filter for the person & face bbox drawings
    filter(roi, image);

    return HAILO_SUCCESS;
}

std::string info_to_str_output_stream(hailort::OutputVStream &stream) {
    std::string result = stream.get_info().name;
    result += " (";
    result += std::to_string(stream.get_info().shape.height);
    result += ", ";
    result += std::to_string(stream.get_info().shape.width);
    result += ", ";
    result += std::to_string(stream.get_info().shape.features);
    result += ")";
    return result;
}

std::string info_to_str_input_stream(hailort::InputVStream &stream) {
    std::string result = stream.get_info().name;
    result += " (";
    result += std::to_string(stream.get_info().shape.height);
    result += ", ";
    result += std::to_string(stream.get_info().shape.width);
    result += ", ";
    result += std::to_string(stream.get_info().shape.features);
    result += ")";
    return result;
}

/**
 * @brief The read thread that process the output vstream and prepare the vector of results
 *
 * @param output The output VStream
 * @param num_of_frames not used
 * @param start_time start time of the write thread
 * @param elapsed_time_S returns the read elapsed time
 * @param output_results returns the read data and vstream info
 * @return hailo_status 
 * @note uses template type T = uint8_t or float32_t
 */
template<typename T>
hailo_status read_all(OutputVStream &output, std::size_t num_of_frames, 
                    std::vector<std::chrono::time_point<std::chrono::system_clock>>& start_time,
                    std::vector<std::chrono::duration<double>>& elapsed_time_s, std::vector<std::pair<std::vector<T>, hailo_vstream_info_t>>& output_results) {
    m.lock();
    std::cout << BOLDCYAN << "-I- Started read thread: " << info_to_str_output_stream(output) << std::endl << RESET;
    m.unlock();    
    
    std::vector<T> data(output.get_frame_size());

    // read the data from the output VStream
    auto status = output.read(MemoryView(data.data(), data.size()) /*MemoryView(data.data(), data.size())*/);

    // calculate elapsed time
    std::chrono::time_point<std::chrono::system_clock> end_t = std::chrono::high_resolution_clock::now();
    elapsed_time_s[0] = end_t - start_time[0];

    // std::cout << YELLOW << std::ends;
    // printf("\r-I-  Recv %lu/%lu",i+1, num_of_frames);
    // std::cout << RESET << std::ends;
    if (HAILO_SUCCESS != status) {
        return status;
    }

    // prepare the output vector
    m2.lock();
    output_results.push_back(std::make_pair<std::vector<uint8_t>, const hailo_vstream_info_t&>(std::move(data), output.get_info()));
    m2.unlock();

    return HAILO_SUCCESS;
}

/**
 * @brief The write thread that process data vector and weites it to the input vstream
 *
 * @param input The input VStream
 * @param num_of_frames not used
 * @param data_array a vector containing all the image data
 * @param start_time returns the read start time
 * @return hailo_status 
 * @note uses template type T = uint8_t or float32_t
 */
template<typename T>
hailo_status write_all(InputVStream &input, std::size_t num_of_frames, std::vector<T>& data_array, 
                        std::vector<std::chrono::time_point<std::chrono::system_clock>>& start_time) {
    m.lock();
    std::cout << BOLDWHITE << "-I- Started write thread: " << info_to_str_input_stream(input) << std::endl << RESET;
    m.unlock();

    // if we are using uint8_t, the we use 1 byte, else it is float then factor will be 4
    int factor = std::is_same<T, uint8_t>::value ? 1 : 4;

    // get the start time
    start_time[0] = std::chrono::high_resolution_clock::now();

    // write the data_array into the input VStream
    auto status = input.write(MemoryView(data_array.data(), data_array.size() * factor));
   
    return status;
 
}

/**
 * @brief Creates a HailoMatrixPtr from an xarray of floats
 *
 * @param xmatrix an xtensor::xarray of floats
 * @return HailoMatrixPtr 
 */
HailoMatrixPtr create_matrix_ptr(xt::xarray<float> &xmatrix)
{
    // allocate and memcpy to a new memory so it points to the right data
    std::vector<float> data(xmatrix.size());
    memcpy(data.data(), xmatrix.data(), sizeof(float) * xmatrix.size());

    return std::make_shared<HailoMatrix>(std::move(data),
                                         xmatrix.shape(0), xmatrix.shape(1), xmatrix.shape(2));
}

/**
 * @brief The infer function - used for 2 networks: yolov5_personface & osnet
 *
 * @param inputs a vector of input VStreams
 * @param outputs a vector of output VStreams
 * @param data_array a vector containing all the image data
 * @param num_of_frames not used
 * @param start_time returns the infer start time
 * @param elapsed_time_t returns the elapsed infer time
 * @param image the image matrix to process
 * @param output_image_name the output image path (not including running counter)
 * @param post_process_type person_face or osnet infer
 * @param currentDetectionInFrame a serial number of a detection in the same frame
 * @return hailo_status 
 * @note uses template type T = uint8_t or float32_tinput_image_name
 */
template<typename IT=float32_t, typename OT=uint8_t>
hailo_status infer(std::vector<InputVStream> &inputs, std::vector<OutputVStream> &outputs, std::vector<IT>& data_array, std::size_t num_of_frames,  
                    std::vector<std::chrono::time_point<std::chrono::system_clock>>& start_time,
                    std::vector<std::chrono::duration<double>>& elapsed_time_s, cv::Mat& image, const cv::String &output_image_name, int post_process_type,
                    int currentDetectInFrame) {

    hailo_status input_status = HAILO_UNINITIALIZED;
    hailo_status output_status = HAILO_UNINITIALIZED;
    std::vector<std::thread> input_threads;
    std::vector<std::thread> output_threads;
    std::vector<std::pair<std::vector<OT>, hailo_vstream_info_t>> output_results;
    std::string output_image_path = "./output_image.png";
    int cropped_image_count = 0;

    // prepare input thread for each input VStream
    for (auto &input: inputs)
        input_threads.push_back( std::thread(
            [&input, &num_of_frames, &data_array, &start_time, &input_status]() 
            { input_status = write_all<IT>(input, num_of_frames, data_array, start_time); }
            ) );
    
    // prepare output thread for each output VStream
    for (auto &output: outputs){
        
        output_threads.push_back( std::thread(
            [&output, &num_of_frames, &start_time, &elapsed_time_s, &output_results, &output_status]() 
            { output_status = read_all<OT>(output, num_of_frames, start_time, elapsed_time_s, output_results); }
            ) );
        
    }

    // join every input thread
    for (auto &in: input_threads)
    {
        in.join();
    }
    
    // join every output thread
    for (auto &out: output_threads)
    {
        out.join();
    }
 
    if ((HAILO_SUCCESS != input_status) || (HAILO_SUCCESS != output_status)) {
        return HAILO_INTERNAL_FAILURE;
    }

    // prepare a region of interest - bounding box defined from (0,0) to (1,1)
    HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));

    xt::xarray<float> embedding;


    // loop thru the output_result vector and add them to the region of intereset
    for (size_t i = 0; i < output_results.size(); i++)
    {     
        roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t *>(&output_results[i].first[0]), output_results[i].second));
    }

    // if it is a person_face infer call
    if (post_process_type == PERSON_FACE_INFER)
    {
        //cv::Mat image = cv::imread(input_image_name, cv::IMREAD_COLOR);
 
        // post process after the read & write thread finished
        output_status = post_process(roi, image);

        auto detections = roi->get_objects_typed(HAILO_DETECTION);

        // for every object in the region of interest
        for (auto object : detections)
        {

            // get the detection type, bounding box coordinates
            HailoObject* obj = object.get();
                
            HailoDetection* detection = (HailoDetection*)obj;
                
            int detectionClassId = detection->get_class_id();
            float xmin = detection->get_bbox().xmin();
            float ymin = detection->get_bbox().ymin();
            float xmax = detection->get_bbox().xmax();
            float ymax = detection->get_bbox().ymax();

            // ensure the xmax & ymax is not bigger than 1
            if (xmax > 1)
                xmax = 1;

            if (ymax > 1)
                ymax = 1;

            // ensure the xmin & ymin is not smaller than 0
            if (xmin < 0)
                xmin = 0;

            if (ymin < 0)
                ymin = 0;

            // get image size
            cv::Size s = image.size();
            float rows = s.height;
            float cols = s.width; 
            
            // if it is person detection (not interested in face detection)
            if (detectionClassId == PERSON_DETECTION)
            {
                num_of_detections_per_frame++;

                // Crop & resize image
                cv::Mat cropped_image = image(cv::Range(ymin*rows,ymax*rows), cv::Range(xmin*cols,xmax*cols));
                cv::resize(cropped_image, cropped_image, cv::Size(128, 256), cv::InterpolationFlags::INTER_AREA);
                
                //Save the cropped Image
                cropped_image_count++;
                cv::String new_output_image_name = output_image_name + std::to_string(cropped_image_count) + ".png";
                
                // write the cropeed image to a png file
                cv::imwrite(new_output_image_name, cropped_image);
                
                // create a unique id for each detection found
                HailoUniqueID id(unique_id_counter);
                HailoUniqueIDPtr idPtr = std::make_shared<HailoUniqueID>(id);
                detection->add_object(idPtr);
                unique_id_counter++;
                
                HailoDetectionPtr detectionPtr = std::make_shared<HailoDetection>(*detection);

                // save the detection in a vector to be used when updating the DB each frame
                m3.lock();
                personDetections.push_back(detectionPtr);
                m3.unlock();                 
            }

            // write the image with the bbox to a file
            // cv::imwrite(output_image_path, image);
        }
    }
    // if it is an osnet infer call 
    else if (post_process_type == OSNET_IFER)
    {    
        if (personDetections.size() != 0)
        {
            // print the 1st 30 places in the output results vector of each detection we found
            // for (size_t i = 0; i < output_results.size(); i++)
            // {
            //     std::cout << BOLDBLUE << "\n\noutput_results[i].first.size()=" << output_results[i].first.size() << std::endl << RESET;
            //     for (size_t j = 0; j < 30; j++)
            //     {
            //         std::cout << BOLDBLUE << "output_results[i].first[j]=" << (uint32_t)(output_results[i].first[j]) << " where i is:" << i << " j is: " << j << std::endl << RESET;
            //     }
            // }

            // loop thru the output_result vector and add them to the region of intereset
            for (size_t i = 0; i < output_results.size(); i++)
            {     
                // Convert the tensor to xarray.
                auto tensor = roi->get_tensor(output_results[i].second.name);
                embedding = common::get_xtensor_float(tensor);

                // vector normalization
                auto normalized_embedding = common::vector_normalization(embedding);
                personDetections[currentDetectInFrame]->add_object(create_matrix_ptr(normalized_embedding));

            } 
        }

        output_status = HAILO_SUCCESS;
    }
    else
    {
        output_status = HAILO_INVALID_ARGUMENT;
    }

    if (HAILO_SUCCESS != output_status) {
        return HAILO_INTERNAL_FAILURE;
    }

    //std::cout << BOLDBLUE << "\n\n-I- Inference finished successfully\n" << std::endl << RESET;
    return output_status;
}

/**
 * @brief prints the hef file name, input & output streams sizes
 *
 * @param hef_file HEF file name
 * @param vstreams a pair of vectors of input & output VStreams
 * @return void 
 */
void print_net_banner(std::string hef_file, std::pair< std::vector<InputVStream>, std::vector<OutputVStream> > &vstreams) {
    std::cout << MAGENTA << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I-  Hailo Network Name                           " << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << "-I-  " << hef_file.substr(0, hef_file.find(".")) << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    for (auto const& value: vstreams.first) {
        std::cout << "-I-  Input shape:  (1, " << value.get_info().shape.height << ", " <<
        value.get_info().shape.width << ", " << value.get_info().shape.features << ")" << std::endl;
    }
    std::cout << "-I-----------------------------------------------" << std::endl;
    for (auto const& value: vstreams.second) {
        std::cout << "-I-  Output shape: (1, " << value.get_info().shape.height << ", " <<
        value.get_info().shape.width << ", " << value.get_info().shape.features << ")" << std::endl;
    }
    std::cout << "-I-----------------------------------------------\n" << std::endl << RESET;
}

/**
 * @brief Configure a device according to HEF file 
 *
 * @param vdevice input device to be configured
 * @param hef_paths a vector of HEF files path
 * @return a vector of ConfiguredNetworkGroup
 */
Expected<std::vector<std::shared_ptr<ConfiguredNetworkGroup>>> configure_hefs(VDevice &vdevice, std::vector<std::string> &hef_paths)
{
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> results;

    // loop thru the HEF files
    for (const auto &path : hef_paths) {
        // create HEF class from the HEF file path
        auto hef_exp = Hef::create(path);
        if (!hef_exp) {
            return make_unexpected(hef_exp.status());
        }
        auto hef = hef_exp.release();

        // configure the VDevice from the HEF file
        auto added_network_groups = vdevice.configure(hef);
        if (!added_network_groups) {
            return make_unexpected(added_network_groups.status());
        }
        // add the network group to the returned vector
        results.insert(results.end(), added_network_groups->begin(),
            added_network_groups->end());
    }
    return results;
}

/**
 * @brief Creates a Vdevice using a scheduler
 *
 * @return a pointer to a VDevice
 */
Expected<std::unique_ptr<VDevice>> create_vdevice()
{
    hailo_vdevice_params_t params;

    // creates default parameters for the vdevice
    auto status = hailo_init_vdevice_params(&params);
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed init vdevice_params, status = " << status << std::endl;
        return make_unexpected(status);
    }

    // we will use a round robin scheduler & 1 device
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    params.device_count = 1;

    // return the created VDevice
    return VDevice::create(params);
}

/**
 * @brief returns a command-line option parameter value 
 *
 * @param argc number of command line parameters
 * @param argv parameters string
 * @param option a string containing the desired parameter
 * @return a string containing the desired command line parameter value
 * @note parameter is set like this: -hef=stam.hef
 */
std::string getCmdOption(int argc, char *argv[], const std::string &option) {
    std::string cmd;

    // loop thru the parameters
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        // get the parameter we want
        if (0 == arg.find(option, 0))
        {
            std::size_t found = arg.find("=", 0) + 1;
            cmd = arg.substr(found, 200);
            return cmd;
        }
    }
    // if no "-num" parameter was specified we set it by default to 100 frames
    if (cmd.empty() && option=="-num=")
        return "100";
    return cmd;
}

/**
 * @brief add a new embedding (matrix) to the RE-ID database 
 *
 * @param global_id the global id of the person to be added to DB
 * @param matrix detection matrix
 * @return none
 * @note supports maximum 100 id's
 */
void add_embedding(uint global_id, HailoMatrixPtr matrix)
{
    global_id--;
    if (reIdDatabase[global_id].size() >= queue_size)
    {
        reIdDatabase[global_id].pop_back();
    }
    reIdDatabase[global_id].insert(reIdDatabase[global_id].begin(), matrix);
}

/**
 * @brief add a new global id to the RE-ID DB 
 *
 * @param matrix detection matrix
 * @return the global id of the added detection
 * @note the global id will be the size of the DB (will be added at the end)
 */
uint add_new_global_id(HailoMatrixPtr matrix)
{
    std::vector<HailoMatrixPtr> queue;
    reIdDatabase.push_back(queue);
    uint global_id = reIdDatabase.size();
    add_embedding(global_id, matrix);
    return global_id;
}

/**
 * @brief multiplies 2 xarrays with the same dimensions
 *
 * @param array1 xtensor xarray
 * @param array2 xtensor xarray
 * @return the dot product of the 2 input arrays
 * @note uses xtensor functions
 */
static float gallery_one_dim_dot_product(xt::xarray<float> array1, xt::xarray<float> array2)
{
    if (array1.dimension() > 1 || array2.dimension() > 1)
    {
        throw std::runtime_error("One of the arrays has more than 1 dimension");
    }
    if (array1.shape(0) != array2.shape(0))
    {
        throw std::runtime_error("Arrays are with different shape");
    }
    return xt::sum(array1 * array2)[0];
}

/**
 * @brief squeeze a matrix using xtensor functions into a xtensor xarray
 *
 * @param matrix a pointer to a HailoMatrix containing the matrix to be squeezed
 * @return a xarray quantization of the input matrix
 */
static xt::xarray<float> gallery_get_xtensor(HailoMatrixPtr matrix)
{
    // Adapt a HailoTensorPtr to an xarray (quantized)
    xt::xarray<float> xtensor = xt::adapt(matrix->get_data().data(), matrix->size(), xt::no_ownership(), matrix->shape());
    return xt::squeeze(xtensor);
}

/**
 * @brief get the (1 - maximum distance between a matrix and a vector of matrices)
 *
 * @param embeddings_queue a vector holding all the matrices of detections we found so far
 * @param matrix the matrix of the detection to be checked against the matrices vector
 * @return the (1 = maximum distance between the input matrix and the vector of matrices)
 * @note uses xtensor functions
 */
static float get_distance(std::vector<HailoMatrixPtr> embeddings_queue, HailoMatrixPtr matrix)
{
    xt::xarray<float> new_embedding = gallery_get_xtensor(matrix);
    float max_thr = 0.0f;
    float thr;

    // loop thru the vector, and multiply them, sum them up and find the maximum distance
    for (HailoMatrixPtr embedding_mat : embeddings_queue)
    {
        xt::xarray<float> embedding = gallery_get_xtensor(embedding_mat);
        thr = gallery_one_dim_dot_product(embedding, new_embedding);
        max_thr = thr > max_thr ? thr : max_thr;
    }

    return 1.0f - max_thr;
}

/**
 * @brief prepare an xarrary containing all the distances between a matrix and all the embedding in the DB
 *
 * @param matrix the matrix of the detection to be checked against the DB of embeddings
 * @return the distances vector
 * @note uses xtensor functions
 */
xt::xarray<float> get_embeddings_distances(HailoMatrixPtr matrix)
{
    std::vector<float> distances;
    for (auto embeddings_queue : reIdDatabase)
    {
        distances.push_back(get_distance(embeddings_queue, matrix));
    }
    return xt::adapt(distances);
}

/**
 * @brief prepare a pair of the global id of a detection with it's distance
 *
 * @param matrix the matrix of the detection 
 * @return a pair containing a global id & it's distance
 * @note uses xtensor functions
 */
std::pair<uint, float> get_closest_global_id(HailoMatrixPtr matrix)
{
    auto distances = get_embeddings_distances(matrix);
    auto global_id = xt::argpartition(distances, 1, xt::xnone())[0];
    return std::pair<uint, float>(global_id+1, distances[global_id]);
}

/**
 * @brief update the DB with the detections found in a frame
 *
 * @param detections a vector of detections find in a frame
 * @param frame_number the frame number (needed for printing only)
 * @return void
 * @note uses xtensor functions
 */
void updateDB(std::vector<HailoDetectionPtr> &detections, int frame_number)
{
    // loop thru all the detections in the input vector
    for (auto detection : detections)
    {
        auto embeddings = detection->get_objects_typed(HAILO_MATRIX);

        // if there is no matrix for the detection, skip to next detection
        if (embeddings.size() == 0)
        {
            std::cout << BOLDRED << "No Embeddings!!!                           " << RESET << std::endl;
            // No HailoMatrix, continue to next detection.
            continue;
        }
        else if (embeddings.size() > 1)
        {
                       
            // More than 1 HailoMatrixPtr is not allowed.
            std::runtime_error("A detection has more than 1 HailoMatrixPtr");
        }

        auto new_embedding = std::dynamic_pointer_cast<HailoMatrix>(embeddings[0]);

        int unique_id = std::dynamic_pointer_cast<HailoUniqueID>(detection->get_objects_typed(HAILO_UNIQUE_ID)[0])->get_id();

        uint global_id;

        // find the unique id in the id's map (if we already have it)
        if (tracking_id_to_global_id.find(unique_id) != tracking_id_to_global_id.end())
        {
            global_id = tracking_id_to_global_id[unique_id];
        }
        // we have a new unique id
        else
        {
            uint closest_global_id;
            float min_distance;

            // If Gallery is empty
            if (reIdDatabase.empty())
            {
                // add the global id to the DB
                global_id = add_new_global_id(new_embedding);
            }
            else
            {
                std::tie(closest_global_id, min_distance) = get_closest_global_id(new_embedding);
                // if smallest distance > threshold -> create new ID
                if (min_distance > similarity_thr)
                {
                    global_id = add_new_global_id(new_embedding);
                }
                else
                {
                    global_id = closest_global_id;
                }
            }

            tracking_id_to_global_id[unique_id] = global_id;
            std::cout << BOLDYELLOW  << "The identified person id is " << global_id << " frame number is " << frame_number << RESET << std::endl;

            // Add global id to detection.
            add_embedding(global_id, new_embedding);
            detection->add_object(std::make_shared<HailoUniqueID>(global_id, GLOBAL_ID));
        }
    }
};

/**
 * @brief the main function 
 *
 * @param argc number of command line parameters
 * @param argv parameters string
 * @return program exit code
 */
int main(int argc, char**argv) {
    std::chrono::duration<double> total_time;
    std::chrono::time_point<std::chrono::system_clock> total_time_start = std::chrono::high_resolution_clock::now();

    // get the program parameters
    std::string hef_file      = getCmdOption(argc, argv, "-hef=");
    std::string re_id_hef_file      = getCmdOption(argc, argv, "-reid=");
    std::size_t num_of_frames = stoi(getCmdOption(argc, argv, "-num="));

    // save the personface & re_id hef files name in a vector
    std::vector<std::string> hef_files;
    hef_files.push_back(hef_file);
    hef_files.push_back(re_id_hef_file);

    std::vector<std::chrono::time_point<std::chrono::system_clock>> start_time_vec(num_of_frames);
    std::vector<std::chrono::duration<double>> elapsed_time_s_vec(num_of_frames);

    ////// LOAD YUV VIDEO AND CONVERT IT TO RGB ////////

    std::string image_path = "./video_images/image"; // image path prefix 
    std::string image_path_suffix = ".png"; // image path suffix
    std::string cropped_image_path = "./cropped_image_"; // cropped image prefix

    cv::Mat image;
    std::vector<float32_t> data_array;

    // create a device
    auto vdevice_exp = create_vdevice();
    if (!vdevice_exp) {
        std::cerr << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();

    // configure the netwrork groups according to the hef files
    auto configured_network_groups_exp = configure_hefs(*vdevice, hef_files);
    if (!configured_network_groups_exp) {
        std::cerr << "Failed to configure HEFs, status = " << configured_network_groups_exp.status() << std::endl;
        return configured_network_groups_exp.status();
    }
    auto configured_network_groups = configured_network_groups_exp.release();

    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_network_group;

    // loop thru all the configured network groups and prepare the vstream params, input & output vstream
    for (auto &network_group : configured_network_groups) {

        auto input_vstream_params = network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!input_vstream_params){
            std::cerr << "-E- Failed make_input_vstream_params " << input_vstream_params.status() << std::endl;
            return input_vstream_params.status();
        }

        auto output_vstream_params = network_group->make_output_vstream_params(true, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!output_vstream_params){
            std::cerr << "-E- Failed make_output_vstream_params " << output_vstream_params.status() << std::endl;
            return output_vstream_params.status();
        }

        auto input_vstreams  = VStreamsBuilder::create_input_vstreams(*network_group, input_vstream_params.value());
        if (!input_vstreams){
            std::cerr << "-E- Failed create_input_vstreams " << output_vstream_params.status() << std::endl;
            return input_vstreams.status();
        }

        auto output_vstreams = VStreamsBuilder::create_output_vstreams(*network_group, output_vstream_params.value());
        if (!input_vstreams or !output_vstreams) {
            std::cerr << "-E- Failed creating input: " << input_vstreams.status() << " output status:" << output_vstreams.status() << std::endl;
            return input_vstreams.status();
        }

        // save the pairs of input & output vstreams to a vector, to be used in the infer function
        vstreams_per_network_group.push_back(std::make_pair(input_vstreams.release(), output_vstreams.release()));
    }

    // print some info of the netowrks we are using
    for (size_t i = 0; i < hef_files.size(); i++) {
        print_net_banner(hef_files[i], vstreams_per_network_group[i]);
    }

    std::string full_image_path;

    // loop thru number of frames (300 in this example)
    for (int i=1; i <= num_of_frames_to_process; i++)
    {
        std::cout << BOLDBLUE << "processing frame number: " << i << RESET << std::endl;

        // build the full image path
        full_image_path = image_path + std::to_string(i) + image_path_suffix;

        // load the image to a cv::Mat
        image = cv::imread(full_image_path, cv::IMREAD_COLOR);

        // convert the color to RGB, resize the image to 640x640 and convert to float32 type
        // this is done cause the personface network require this as input
        cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
        cv::resize(image, image, cv::Size(640, 640), cv::InterpolationFlags::INTER_AREA);
        image.convertTo(image, CV_32F_TYPE, 1.0);

        // build a vector of float32 from the image
        data_array.assign((float32_t*)image.data, (float32_t*)image.data + image.total()*image.channels());

        num_of_detections_per_frame = 0;

        // call the 1st infer with the 1st network's vdevice pair (the personface network)
        auto status  = infer(vstreams_per_network_group[0].first, vstreams_per_network_group[0].second, data_array, 
                            num_of_frames, start_time_vec, elapsed_time_s_vec,image,cropped_image_path,PERSON_FACE_INFER,0);

    
        double total_elapsed_time = 0.0;
        
        // calculate the elapsed time
        for (size_t i = 0; i < num_of_frames; i++){
            total_elapsed_time += elapsed_time_s_vec[i].count();
        }
        
        print_inference_statistics(num_of_frames, total_elapsed_time);

        // calculate total time since main started
        std::chrono::time_point<std::chrono::system_clock> total_time_end = std::chrono::high_resolution_clock::now();
        total_time = total_time_end - total_time_start;

        if (HAILO_SUCCESS != status) {
            std::cerr << "-E- Inference failed "  << status << std::endl;
            return status;
        }

        std::cout << BOLDBLUE << "-I- Total inference run time: " << (double)total_time.count() << " sec" << RESET << std::endl;

        cv::Mat new_image;
        std::vector<float32_t> new_data_array;
        cv::String cropped_image_name;
        
        // loop thru all the detections we got in the 1st infer
        for (int j=1; j<(num_of_detections_per_frame+1); j++)
        {
            // prepare the cropped image full path name
            // the cropeed image contains a bounding box of 256x128 pixels
            cropped_image_name = cropped_image_path + std::to_string(j) + ".png";

            // read the cropped image and convert it to uint_8 type (needed for the re-id network)
            new_image = cv::imread(cropped_image_name, cv::IMREAD_COLOR);
            new_image.convertTo(new_image, CV_8U_TYPE, 1.0);

            // build a vector of uint8 from the image
            new_data_array.assign((uint8_t*)new_image.data, (uint8_t*)new_image.data + new_image.total()*new_image.channels());

            // call the 2nd infer with the 2nd network's vdevice pair (the osnet network)
            status  = infer(vstreams_per_network_group[1].first, vstreams_per_network_group[1].second, new_data_array, 
                                num_of_frames, start_time_vec, elapsed_time_s_vec,new_image,"new_output_image.png",OSNET_IFER,j-1);

            if (HAILO_SUCCESS != status) {
                std::cerr << "-E- Inference failed "  << status << std::endl;
                return status;
            }
        }

        // calculate the elapsed time
        for (size_t k = 0; k < num_of_frames; k++){
            total_elapsed_time += elapsed_time_s_vec[k].count();
        }
        
        print_inference_statistics(num_of_frames, total_elapsed_time);

        // calculate the total processing time
        total_time_end = std::chrono::high_resolution_clock::now();
        total_time = total_time_end - total_time_start;

        // update the DB with the new detections found in the frame
        updateDB(personDetections,i);

        // clear the detections vector for next iteration
        personDetections.clear();

        // end of 2nd Infer
    }

    std::cout << BOLDBLUE << "-I- Total inference run time: " << (double)total_time.count() << " sec" << RESET << std::endl;

    return HAILO_SUCCESS;
}
