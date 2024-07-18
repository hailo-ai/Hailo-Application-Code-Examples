#include "media_library/yolov8.hpp"
#include "media_library/yolov8_postprocess.hpp"
#include <tl/expected.hpp>
#include <hailo/hailort.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/vstream.hpp>

#include <iostream>

#include "media_library/common/common.h"


#include "media_library/common/hailo_objects.hpp"
#include "media_library/common/hailomat.hpp"


#include <iostream>
#include <chrono>
#include <mutex>
#include <future>

#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;

using namespace hailort;


// post process of yolov8
hailo_status Yolov8::post_process(std::vector<std::shared_ptr<FeatureData>> &features){
    auto status = HAILO_SUCCESS;
    while(!stopSignal){
        // Make sure the queue is not empty and get the first element
        // The queue contains buffers after read
        while(read_pp_queue.empty() && !stopSignal){
            sleep(0.01);
        }   
        if(stopSignal){
            return HAILO_SUCCESS;
        }
        hailo_media_library_buffer *image;
        read_pp_queue.pop(image);

        // get the detections from features
        std::sort(features.begin(), features.end(), &FeatureData::sort_tensors_by_size);
        HailoNV12Mat frame = HailoNV12Mat(NULL, height, width, width, width, 1, 1, image->hailo_pix_buffer->planes[0].userptr, image->hailo_pix_buffer->planes[1].userptr);

        HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        
        for (auto& feature : features) {
            roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(feature->m_buffers.get_read_buffer().data()), feature->m_vstream_info));
        }
        filter(roi);
        for (auto &feature : features) {
            feature->m_buffers.release_read_buffer();
        } 

        std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

        // draw the detections to the image
        for (auto &detection : detections) {
            HailoBBox bbox = detection->get_bbox();
            cv::Rect rect;
            if(bbox.xmin() >= 0 && bbox.xmax() <= 1 && bbox.xmax() > bbox.xmin() && bbox.ymin() >= 0 && bbox.ymax() <= 1 && bbox.ymax() > bbox.ymin()){ 
                rect.x = bbox.xmin() * float(width);
                rect.y = bbox.ymin() * float(height);
                rect.width = bbox.width() * float(width);
                rect.height = bbox.height() * float(height) ;
            }
            frame.draw_rectangle(rect, cv::Scalar(0, 0, 255));
            std::cout << "Frame " << ", Detection: " << detection->get_label() << ", Confidence: " << std::fixed << std::setprecision(2) << detection->get_confidence() * 100.0 << "%" << std::endl;
        }
        // assign the image with the detections to the image without the detections
        // NV12 has 2 dimensions 
        image->hailo_pix_buffer->planes[0].userptr = frame.get_matrices()[0].data;
        image->hailo_pix_buffer->planes[1].userptr = frame.get_matrices()[1].data;
        pp_output_queue.push(image);
    }
    return status;
}

// read from device
hailo_status Yolov8::read(std::vector<std::shared_ptr<FeatureData>> features){
    // Make sure the queue is not empty and get the first element
    // The queue contains buffers after write
    // This function doesn't do anything with the buffer from the queue.
    // The queue is here for synchronization of the threads
    while (!stopSignal)
    {
    std::cout << "start read" << std::endl;
    while(write_read_queue.empty() && !stopSignal){
        sleep(0.01);
    }   
    if(stopSignal){
        return HAILO_SUCCESS;
    }
    hailo_media_library_buffer *image;
    write_read_queue.pop(image);
    // read from the device for each output stream 
    for (size_t j = 0; j < (size_t)vstreams.get()->second.size(); j++) {
        auto& buffer = features[j]->m_buffers.get_write_buffer();
        hailo_status status = vstreams.get()->second[j].read(MemoryView(buffer.data(), buffer.size()));
        features[j]->m_buffers.release_write_buffer();
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed reading with status = " <<  status << std::endl;
            return status;
        }
    }
    read_pp_queue.push(image);
    }

    return HAILO_SUCCESS;
}

// create hailo_pix_buffer_t from the hailo_media_library_buffer
void create_hailo_pix_buffer(hailo_media_library_buffer* buffer, hailo_pix_buffer_t* hailo_pix_buffer){
    hailo_pix_buffer->number_of_planes = buffer->get_num_of_planes();
    for(int i = 0; i < hailo_pix_buffer->number_of_planes; i++){
        hailo_pix_buffer->planes[i].bytes_used = buffer->hailo_pix_buffer->planes[i].bytesused;
        hailo_pix_buffer->planes[i].plane_size = buffer->hailo_pix_buffer->planes[i].bytesperline;
        hailo_pix_buffer->planes[i].user_ptr = buffer->hailo_pix_buffer->planes[i].userptr;
    }
}

// write buffers to the hailo 15 device
hailo_status Yolov8::write() {                    
    hailo_status status = HAILO_SUCCESS;
    // Make sure the queue is not empty and get the first element
    // The queue contains buffers after vision preproc
    while(!stopSignal){

        // make sure the queue is not empty and get the first element
        while (input_write_queue.empty() && !stopSignal){
            sleep(0.01);
        }
        if(stopSignal){
            return HAILO_SUCCESS;
        }
        // create hailo_pix_buffer_t from the buffer
        hailo_media_library_buffer* buffer;
        input_write_queue.pop(buffer);
        hailo_pix_buffer_t *hailo_pix_buffer = new hailo_pix_buffer_t();
        create_hailo_pix_buffer(buffer, hailo_pix_buffer);

        // write the image to the device
        status = vstreams.get()->first[0].write(*hailo_pix_buffer);
        if (HAILO_SUCCESS != status){
            std::cerr << "cannot write to device" << std::endl;
            return status;
        }
        delete hailo_pix_buffer;
        write_read_queue.push(buffer);
    }
    return HAILO_SUCCESS;
}

// features is the shared memory allocated for inference data
hailo_status create_feature(hailo_vstream_info_t vstream_info, size_t output_frame_size, std::shared_ptr<FeatureData> &feature) {
    feature = std::make_shared<FeatureData>(static_cast<uint32_t>(output_frame_size), vstream_info.quant_info.qp_zp,
        vstream_info.quant_info.qp_scale, vstream_info.shape.width, vstream_info);

    return HAILO_SUCCESS;
}
// print input and output vstreams
void print_net_banner(std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> &vstreams) {
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-  Network  Name                                     " << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto const& value: vstreams.first) {
        std::cout << MAGENTA << "-I-  IN:  " << value.name() <<std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto const& value: vstreams.second) {
        std::cout << MAGENTA << "-I-  OUT: " << value.name() <<std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}

// create hef and configure network group 
Expected<std::shared_ptr<ConfiguredNetworkGroup>> configure_network_group(VDevice &vdevice, const std::string yolov_hef)
{
    auto hef_exp = Hef::create(yolov_hef);
    if (!hef_exp) {
        return make_unexpected(hef_exp.status());
    }
    auto hef = hef_exp.release();

    auto configure_params = vdevice.create_configure_params(hef);
    if (!configure_params) {
        return make_unexpected(configure_params.status());
    }

    auto network_groups = vdevice.configure(hef, configure_params.value());
    if (!network_groups) {
        return make_unexpected(network_groups.status());
    }

    if (1 != network_groups->size()) {
        std::cout << "Invalid amount of network groups" << std::endl;
        return make_unexpected(HAILO_INTERNAL_FAILURE);
    }

    return std::move(network_groups->at(0));
}

// Create all the shared memory for the threads, and the connection to the Hailo device
Yolov8::Yolov8(std::string hef, int width, int height): input_write_queue(kQueueCapacity), read_pp_queue(kQueueCapacity), write_read_queue(kQueueCapacity), pp_output_queue(kQueueCapacity), stopSignal(false), width(width), height(height){
    //Create device
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cout << "Failed create vdevice, status = " << vdevice_exp.status() << std::endl;
    }
    vdevice = vdevice_exp.release();

    //Create network group
    auto network_group_exp = configure_network_group(*vdevice, hef);
    if (!network_group_exp) {
        std::cout << "Failed to configure network group " << hef << std::endl;
    }
    network_group = network_group_exp.release();
    auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, QUANTIZED, FORMAT_TYPE);
    if (!vstreams_exp) {
        std::cout << "Failed creating vstreams " << vstreams_exp.status() << std::endl;
    }
    vstreams = std::make_shared<std::pair<std::vector<hailort::InputVStream>,std::vector<hailort::OutputVStream>>>(vstreams_exp.release());
    print_net_banner(*(vstreams.get()));
    auto output_vstreams_size = vstreams.get()->second.size();

    // features is the shared memory allocated for inference data
    features.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++) {
        std::shared_ptr<FeatureData> feature(nullptr);
        auto status = create_feature(vstreams.get()->second[i].get_info(), vstreams.get()->second[i].get_frame_size(), feature);
        if (HAILO_SUCCESS != status) {
            std::cout << "Failed creating feature with status = " << status << std::endl;
            break;
        }
    features.emplace_back(feature);
    }
}

hailo_status Yolov8::run() {
    hailo_status status = HAILO_UNINITIALIZED;

    // Create the write thread
    write_thread = std::async(&Yolov8::write, this);

    // Create the read thread
    read_thread = std::async(&Yolov8::read, this, std::ref(features));

    // Create the postprocessing thread
    pp_thread = std::async(&Yolov8::post_process, this, std::ref(features));

    return status;
}

void Yolov8::add_frame(hailo_media_library_buffer& buffer){
    input_write_queue.push(&buffer);
}

// stop the threads
void Yolov8::stop(){
    stopSignal =true;
    write_thread.wait();
    read_thread.wait();
    pp_thread.wait();
}

boost::lockfree::queue<hailo_media_library_buffer*>* Yolov8::get_queue(){
    return &pp_output_queue;
}