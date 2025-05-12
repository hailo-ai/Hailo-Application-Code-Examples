#pragma once

#include "stage.hpp"
#include "buffer.hpp"
#include "media_library/dsp_utils.hpp"
#include "hailo_common.hpp"
#include <algorithm>


#define DETECTOR_WIDTH 1920
#define DETECTOR_HEIGHT 1080

#define CROP_MAX_WIDTH 3840
#define CROP_MAX_HEIGHT 2160

/**
 * @brief Base class for DSP crop stages, responsible for handling common cropping and resizing operations.
 */
class DspBaseCropStage : public ConnectedStage
{
protected:
    MediaLibraryBufferPoolPtr m_buffer_pool; /**< Buffer pool for managing media library buffers */
    int m_output_pool_size; /**< Size of the output buffer pool */
    int m_input_width;  /**< Width of the input data */
    int m_input_height; /**< Height of the input data */
    int m_output_width; /**< Width of the output data */
    int m_output_hight; /**< Height of the output data */

    std::string m_main_subscriber; /**< Name of the main subscriber */
    std::string m_sub_subscriber; /**< Name of the sub-subscriber */

public:
    /**
     * @brief Constructor to initialize the stage with specified parameters.
     * @param name Name of the stage.
     * @param output_pool_size Size of the output buffer pool.
     * @param input_width Width of the input data.
     * @param input_height Height of the input data.
     * @param output_width Width of the output data.
     * @param output_height Height of the output data.
     * @param main_sub_name Name of the main subscriber.
     * @param sub_sub_name Name of the sub-subscriber.
     * @param queue_size Size of the queue.
     * @param leaky Boolean flag for leaky behavior.
     * @param print_fps Boolean flag for printing FPS.
     */
    DspBaseCropStage(std::string name, int output_pool_size, int input_width, int input_height, 
                    int output_width, int output_height,
                    std::string main_sub_name, std::string sub_sub_name,
                    size_t queue_size, bool leaky = false, bool print_fps=false) : ConnectedStage(name, queue_size, leaky, print_fps),
                                          m_output_pool_size(output_pool_size), m_input_width(input_width), m_input_height(input_height), 
                                          m_output_width(output_width), m_output_hight(output_height),
                                          m_main_subscriber(main_sub_name), m_sub_subscriber(sub_sub_name) {}

    
    /**
     * @brief Prepares cropping dimensions for a single bounding box.
     * @param bbox Bounding box for cropping.
     * @param crop_resize_dims Vector to store crop resize dimensions.
     */
    virtual void prepare_single_crop_dim(HailoBBox bbox, std::vector<dsp_crop_api_t> &crop_resize_dims)
    {
        dsp_crop_api_t crop_resize_dim = {
            .start_x = (size_t)std::clamp((bbox.xmin() * m_input_width), (float)0.0, ((float)m_input_width) - (float)1.0), 
            .start_y = (size_t)std::clamp((bbox.ymin() * m_input_height), (float)0.0, ((float)m_input_height) - (float)1.0),
            .end_x = (size_t)std::clamp(((bbox.xmin() * m_input_width) + (bbox.width() * m_input_width)), (float)1.0, (float)m_input_width),
            .end_y = (size_t)std::clamp(((bbox.ymin() * m_input_height) + (bbox.height() * m_input_height)), (float)1.0, (float)m_input_height),
        };

        /* DSP API can't get dimension that are not even */
        if (crop_resize_dim.start_x % 2  != 0)
            crop_resize_dim.start_x += 1;
        
        if (crop_resize_dim.start_y % 2  != 0)
            crop_resize_dim.start_y += 1;
        
        if (crop_resize_dim.end_x % 2  != 0)
            crop_resize_dim.end_x += 1;
        
        if (crop_resize_dim.end_y % 2  != 0)
            crop_resize_dim.end_y += 1;

        crop_resize_dims.push_back(crop_resize_dim);
    }
    
    /**
     * @brief Prepares crop dimensions for the input buffer.
     * @param input_buffer Input buffer.
     * @param crop_resize_dims Vector to store crop dimensions.
     */
    virtual void prepare_crops(BufferPtr input_buffer, std::vector<dsp_crop_api_t> &crop_resize_dims) = 0;
    
    /**
     * @brief Gets the bounding box for a specific crop.
     * @param index Index of the crop.
     * @return Bounding box of the crop.
     */
    virtual HailoBBox get_crop_bbox(int index) = 0;

    /**
     * @brief Performs post-processing after cropping.
     * @param input_buffer Input buffer.
     */
    virtual void post_crop(BufferPtr input_buffer) {}

    /**
     * @brief Performs pre-processing before cropping.
     * @param input_buffer Input buffer.
     */
    virtual void pre_crop(BufferPtr input_buffer) {}

    /**
     * @brief Gets the ROI for a specific crop.
     * @param index Index of the crop.
     * @return ROI of the crop.
     */
    virtual HailoROIPtr get_crop_roi(int index) { return nullptr; }

    /**
     * @brief Processes the data buffer, performing cropping and resizing.
     * @param data Data buffer.
     * @return Status of the operation.
     */
    AppStatus process(BufferPtr data) override
    {
        dsp_status status;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::vector<dsp_crop_resize_params_t> crops_params;
        std::vector<dsp_crop_api_t> crop_resize_dims;
        std::vector<HailoMediaLibraryBufferPtr> cropped_buffers;
        std::vector<hailo_dsp_buffer_data_t> output_dsp_buffers;
        prepare_crops(data, crop_resize_dims);

        std::size_t num_crops_allowed = std::min(crop_resize_dims.size(), (std::size_t)m_output_pool_size);

        crops_params.reserve(num_crops_allowed);
        output_dsp_buffers.reserve(num_crops_allowed);

        for (std::size_t i = 0; i < num_crops_allowed; ++i)
        {
            auto &dims = crop_resize_dims[i];

            HailoMediaLibraryBufferPtr cropped_buffer = std::make_shared<hailo_media_library_buffer>();
            if (m_buffer_pool->acquire_buffer(cropped_buffer) != MEDIA_LIBRARY_SUCCESS)
            {
                std::cerr << "Failed to acquire buffer " << m_stage_name <<std::endl;
                return AppStatus::DSP_OPERATION_ERROR;
            }

            output_dsp_buffers.emplace_back(std::move(cropped_buffer->buffer_data->As<hailo_dsp_buffer_data_t>()));
            dsp_crop_resize_params_t crop_resize_params = {
                .crop = &dims,
            };

            crop_resize_params.dst[0] = &output_dsp_buffers[i].properties;
            crops_params.emplace_back(std::move(crop_resize_params));
            cropped_buffers.emplace_back(cropped_buffer);
        }

        HailoMediaLibraryBufferPtr input_buffer = data->get_buffer();
        hailo_dsp_buffer_data_t in_buffer_data = input_buffer->buffer_data->As<hailo_dsp_buffer_data_t>();
        dsp_multi_crop_resize_params_t multi_crop_resize_params = {
            .src = &in_buffer_data.properties,
            .crop_resize_params = crops_params.data(),
            .crop_resize_params_count = num_crops_allowed,
            .interpolation = INTERPOLATION_TYPE_BILINEAR,
        };

        status = dsp_utils::perform_dsp_multi_resize(&multi_crop_resize_params);
        if (status != DSP_SUCCESS) 
        {
            std::cerr << "Failed to perform dsp multi resize" << std::endl;
            return AppStatus::DSP_OPERATION_ERROR;
        }

        CroppingMetadataPtr cropping_meta = std::make_shared<CroppingMetadata>(cropped_buffers.size());
        data->add_metadata(cropping_meta);
        
        data->add_time_stamp(m_stage_name);
        set_duration(data);
        send_to_specific_subsciber(m_main_subscriber, data);

        for (std::size_t i = 0; i < cropped_buffers.size(); ++i)
        {
            HailoROIPtr roi = get_crop_roi(i);
            BufferPtr cropped_buffer_ptr = std::make_shared<Buffer>(cropped_buffers[i], roi);

            // Set the ROI of the cropped buffer to the scale of the parent ROI
            // Note, this will make overlay incorrect if the bboxes are not flattened
            cropped_buffer_ptr->get_roi()->set_scaling_bbox(get_crop_bbox(i));
            cropped_buffer_ptr->add_time_stamp(m_stage_name+"_"+ std::to_string(i));

            send_to_specific_subsciber(m_sub_subscriber, cropped_buffer_ptr);
        }

        post_crop(data);
        
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        if (m_print_fps)
        {
            std::cout << "Crop and resize time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() 
                      << "[microseconds]" << "Number of crops: " << crop_resize_dims.size() << std::endl;
        }

        return AppStatus::SUCCESS;
    }
};

/**
 * @brief Derived class of DspBaseCropStage for handling specific cropping logic based on predefined tiles.
 */
class TillingCropStage : public DspBaseCropStage
{
private:
    /**< Predefined bounding boxes for tiles */
    std::vector<HailoBBox> m_bbox_tiles;
    std::vector<HailoTileROIPtr> m_fhd_tiles; /**< Tile ROI pointers for FHD tiles */

public:
    /**
     * @brief Constructor to initialize the stage with specified parameters.
     * @param name Name of the stage.
     * @param output_pool_size Size of the output buffer pool.
     * @param input_width Width of the input data.
     * @param input_height Height of the input data.
     * @param output_width Width of the output data.
     * @param output_height Height of the output data.
     * @param main_sub_name Name of the main subscriber.
     * @param sub_sub_name Name of the sub-subscriber.
     * @param queue_size Size of the queue.
     * @param leaky Boolean flag for leaky behavior.
     * @param print_fps Boolean flag for printing FPS.
     */
    TillingCropStage(std::string name, int output_pool_size, int input_width, int input_height, 
                int output_width, int output_height,
                std::string main_sub_name, std::string sub_sub_name, std::vector<HailoBBox> bbox_tiles,
                size_t queue_size, bool leaky=false, bool print_fps=false) : DspBaseCropStage(name, output_pool_size, input_width, input_height,
                                                                        output_width, output_height,
                                                                        main_sub_name, sub_sub_name,
                                                                        queue_size, leaky, print_fps), m_bbox_tiles(bbox_tiles) {}
    /**
     * @brief Initializes the buffer pool and tile ROIs.
     * @return Status of the operation.
     */
    AppStatus init() override
    {
        auto bytes_per_line = dsp_utils::get_dsp_desired_stride_from_width(m_output_width);
        m_buffer_pool = std::make_shared<MediaLibraryBufferPool>(m_output_width, m_output_hight, HAILO_FORMAT_NV12,
                                                                 m_output_pool_size, HAILO_MEMORY_TYPE_DMABUF, bytes_per_line, "tilling_buffer_pool");
           
        if (m_buffer_pool->init() != MEDIA_LIBRARY_SUCCESS)
        {
            return AppStatus::DSP_OPERATION_ERROR;
        }
        
        /* Create the HailoTileROI objects and the buffer pools we will have pool per tile */
        for (std::size_t i = 0; i < m_bbox_tiles.size(); ++i)
        {
            const auto &tile_bbox = m_bbox_tiles[i]; 
            HailoTileROIPtr tile = std::make_shared<HailoTileROI>(tile_bbox, 0, 0, 0, 0, SINGLE_SCALE);
            m_fhd_tiles.push_back(tile);
        }
       
        return AppStatus::SUCCESS;
    }

    /**
     * @brief Prepares crop dimensions based on tiles.
     * @param input_buffer Input buffer.
     * @param crop_resize_dims Vector to store crop dimensions.
     */
    void prepare_crops(BufferPtr input_buffer, std::vector<dsp_crop_api_t> &crop_resize_dims) override
    {
        for (auto &tile : m_fhd_tiles)
        {
            HailoBBox bbox = tile->get_bbox();
            prepare_single_crop_dim(bbox, crop_resize_dims);
        }
    }

    /**
     * @brief Gets the bounding box for a specific tile.
     * @param index Index of the tile.
     * @return Bounding box of the tile.
     */
    HailoBBox get_crop_bbox(int index) override
    {
        try {
            return m_fhd_tiles.at(index)->get_bbox();
        } catch (const std::out_of_range& e) {
             std::cerr << "Tilling index " << index << " is out of bounds: " << e.what() << std::endl;
             throw;
        }
    }

};

/**
 * @brief Derived class of DspBaseCropStage for handling cropping based on detected bounding boxes.
 */
class BBoxCropStage : public DspBaseCropStage
{
private:
    std::vector<HailoBBox> m_detection_crops_bbox; /**< Bounding boxes for detected crops */
    std::vector<HailoROIPtr> m_detection_rois; /**< ROI pointers for detections */
    std::string m_target_label; /**< Target label for filtering detections */
public:

    /**
     * @brief Constructor to initialize the stage with specified parameters.
     * @param name Name of the stage.
     * @param output_pool_size Size of the output buffer pool.
     * @param input_width Width of the input data.
     * @param input_height Height of the input data.
     * @param output_width Width of the output data.
     * @param output_height Height of the output data.
     * @param main_sub_name Name of the main subscriber.
     * @param sub_sub_name Name of the sub-subscriber.
     * @param label Target label for filtering detections.
     * @param queue_size Size of the queue.
     * @param leaky Boolean flag for leaky behavior.
     * @param print_fps Boolean flag for printing FPS.
     */
    BBoxCropStage(std::string name, int output_pool_size, int input_width, int input_height, 
                int output_width, int output_height,
                std::string main_sub_name, std::string sub_sub_name, std::string label,
                size_t queue_size, bool leaky=false, bool print_fps=false) : DspBaseCropStage(name, output_pool_size, input_width, input_height,
                                                                        output_width, output_height,
                                                                        main_sub_name, sub_sub_name,
                                                                        queue_size, leaky), m_target_label(label) { }

    /**
     * @brief Initializes the buffer pool.
     * @return Status of the operation.
     */
    AppStatus init() override
    {
        auto bytes_per_line = dsp_utils::get_dsp_desired_stride_from_width(m_output_width);
        m_buffer_pool = std::make_shared<MediaLibraryBufferPool>(m_output_width, m_output_hight, HAILO_FORMAT_NV12,
                                                                 m_output_pool_size, HAILO_MEMORY_TYPE_DMABUF, bytes_per_line, "detection_buffer_pool");
           
        if (m_buffer_pool->init() != MEDIA_LIBRARY_SUCCESS)
        {
            return AppStatus::DSP_OPERATION_ERROR;
        }

        return AppStatus::SUCCESS;
    }

    /**
     * @brief Prepares crop dimensions based on detected bounding boxes.
     * @param input_buffer Input buffer.
     * @param crop_resize_dims Vector to store crop dimensions.
     */
    void prepare_crops(BufferPtr input_buffer, std::vector<dsp_crop_api_t> &crop_resize_dims) override
    {
        HailoROIPtr roi = input_buffer->get_roi();
        
        for (auto detection : hailo_common::get_hailo_detections(roi))
        {
            if (detection->get_label() == m_target_label)
            {
                auto detection_bbox = detection->get_bbox();
    
                m_detection_crops_bbox.push_back(detection_bbox);
                m_detection_rois.push_back(detection);
                
                prepare_single_crop_dim(detection_bbox, crop_resize_dims);
                
            }
        }
    }
    
    /**
     * @brief Gets the bounding box for a specific detection.
     * @param index Index of the detection.
     * @return Bounding box of the detection.
     */
    HailoBBox get_crop_bbox(int index) override
    {
        try {
            return m_detection_crops_bbox.at(index);
        } catch (const std::out_of_range& e) {
            std::cerr << "Cropped index " << index << " is out of bounds: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * @brief Gets the ROI for a specific detection.
     * @param index Index of the detection.
     * @return ROI of the detection.
     */
    HailoROIPtr get_crop_roi(int index) override
    {
        try {
            return m_detection_rois.at(index);
        } catch (const std::out_of_range& e) {
            std::cerr << "ROI index " << index << " is out of bounds: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * @brief Clears detection data after cropping.
     * @param input_buffer Input buffer.
     */
    void post_crop(BufferPtr input_buffer) override
    {
        m_detection_crops_bbox.clear();
        m_detection_rois.clear();
    }

};
