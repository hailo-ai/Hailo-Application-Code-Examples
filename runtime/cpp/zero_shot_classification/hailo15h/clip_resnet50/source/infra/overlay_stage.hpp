#pragma once
#include "stage.hpp"
#include "buffer.hpp"
#include "queue.hpp"

#include "overlay_native.hpp"

/**
 * @struct HailoOverlay
 * @brief Structure containing configuration parameters for overlay.
 */
struct HailoOverlay
{
    int line_thickness;                 /**< Line thickness for overlay. */
    int font_thickness;                 /**< Font thickness for overlay. */
    float landmark_point_radius;        /**< Radius for landmark points. */
    bool face_blur;                     /**< Enable or disable face blur. */
    bool show_confidence;               /**< Enable or disable confidence display. */
    bool local_gallery;                 /**< Enable or disable local gallery usage. */
    uint mask_overlay_n_threads;        /**< Number of threads for mask overlay. */
};

/**
 * @class OverlayStage
 * @brief Class responsible for handling overlay stage that will do the drawing.
 */
class OverlayStage : public ConnectedStage
{
private:
    HailoOverlay m_hailooverlay_info;   /**< Overlay configuration parameters. */
    
public:
    /**
     * @brief Constructor for OverlayStage.
     * @param name The name of the stage.
     * @param queue_size Size of the queue for this stage.
     * @param leaky Indicates if the queue is leaky.
     * @param print_fps Flag to enable or disable printing FPS information.
     */
    OverlayStage(std::string name, size_t queue_size=5, bool leaky=false, bool print_fps=false) : 
        ConnectedStage(name, queue_size, leaky, print_fps){}

    /**
     * @brief Initialize the overlay stage.
     * @return Status of the initialization.
     */
    AppStatus init() override
    {
        /* Set overlay default values */
        m_hailooverlay_info.line_thickness = 1;
        m_hailooverlay_info.font_thickness = 1;
        m_hailooverlay_info.face_blur = false;
        m_hailooverlay_info.show_confidence = true;
        m_hailooverlay_info.local_gallery = false;
        m_hailooverlay_info.landmark_point_radius = 3;
        m_hailooverlay_info.mask_overlay_n_threads = 0;
        return AppStatus::SUCCESS;
    }

    /**
     * @brief Deinitialize the overlay stage.
     * @return Status of the deinitialization.
     */
    AppStatus deinit() override
    {
        return AppStatus::SUCCESS;
    }

    /**
     * @brief Process the given data buffer and apply overlay.
     * @param data The data buffer to process.
     * @return Status of the processing.
     */
    AppStatus process(BufferPtr data)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        overlay_status_t ret = OVERLAY_STATUS_UNINITIALIZED;

        std::shared_ptr<HailoMat> hmat = std::make_shared<HailoNV12Mat>((uint8_t*)data->get_buffer()->get_plane_ptr(0),
                                              data->get_buffer()->buffer_data->height,
                                              data->get_buffer()->buffer_data->width,
                                              data->get_buffer()->get_plane_stride(0),
                                              data->get_buffer()->get_plane_stride(1),
                                              m_hailooverlay_info.line_thickness,
                                              m_hailooverlay_info.font_thickness,
                                              (uint8_t*)data->get_buffer()->get_plane_ptr(0),
                                              (uint8_t*)data->get_buffer()->get_plane_ptr(1));

        if (hmat)
        {
            if (DmaMemoryAllocator::get_instance().dmabuf_sync_start(data->get_buffer()->get_plane_ptr(0)) != MEDIA_LIBRARY_SUCCESS)
                    return AppStatus::DMA_ERROR;
            if (DmaMemoryAllocator::get_instance().dmabuf_sync_start(data->get_buffer()->get_plane_ptr(1)) != MEDIA_LIBRARY_SUCCESS)
                    return AppStatus::DMA_ERROR;
            // Blur faces if face-blur is activated.
            if (m_hailooverlay_info.face_blur)
            {
                face_blur(*hmat.get(), data->get_roi());
            }
            // Draw all results of the given roi on mat.
            ret = draw_all(*hmat.get(), data->get_roi(), m_hailooverlay_info.landmark_point_radius, m_hailooverlay_info.show_confidence, m_hailooverlay_info.local_gallery, m_hailooverlay_info.mask_overlay_n_threads);
            if (ret != OVERLAY_STATUS_OK)
            {
                std::cerr << " Overlay failure draw_all failed, status = " << ret << std::endl;
            }
            if (DmaMemoryAllocator::get_instance().dmabuf_sync_end(data->get_buffer()->get_plane_ptr(0)) != MEDIA_LIBRARY_SUCCESS)
                    return AppStatus::DMA_ERROR;
            if (DmaMemoryAllocator::get_instance().dmabuf_sync_end(data->get_buffer()->get_plane_ptr(1)) != MEDIA_LIBRARY_SUCCESS)
                    return AppStatus::DMA_ERROR;
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (m_print_fps)
        {
            std::cout << "Overlay time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;
        }
        data->add_time_stamp(m_stage_name);
        set_duration(data);
        send_to_subscribers(data);

        return AppStatus::SUCCESS;
    }
};
