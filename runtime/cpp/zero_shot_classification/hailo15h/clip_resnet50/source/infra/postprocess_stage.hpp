#pragma once

// General includes
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <dlfcn.h>
#include <vector>

// HailoRT includes
#include "hailo/hailort.hpp"

// Tappas includes
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

// Media library includes
#include "media_library/media_library_types.hpp"

// Infra includes
#include "buffer.hpp"
#include "queue.hpp"
#include "stage.hpp"

// Defines
#define DEFAULT_FUNC_NAME "filter"
#define INIT_FUNC_NAME "init"
#define FREE_FUNC_NAME "free_resources"

/**
 * @brief Class representing a post-processing stage in the connected stage pipeline.
 * 
 * This class is responsible for loading a shared object library, initializing it, and
 * applying post-processing functions to the data.
 */
class PostprocessStage : public ConnectedStage
{
private:
    // Library info
    std::string m_so_path;          ///< Path to the shared object file.
    std::string m_config_path;      ///< Path to the configuration file.
    std::string m_function_name;    ///< Name of the function to be executed from the shared object file.

    // Loaded libraries and params
    void *m_loaded_lib;             ///< Handle to the loaded shared object library.
    void *m_params;                 ///< Parameters for the post-processing function.

    // Function handlers
    void (*m_handler)(HailoROIPtr, void *);             ///< Function pointer to the post-processing function with parameters.
    void (*m_handler_no_config)(HailoROIPtr);           ///< Function pointer to the post-processing function without parameters.
    std::chrono::steady_clock::time_point m_last_time;  ///< Timestamp of the last processed frame.

public:
    /**
     * @brief Construct a new Postprocess Stage object.
     * 
     * @param name Name of the stage.
     * @param so_path Path to the shared object file.
     * @param function_name Name of the function to be executed from the shared object file.
     * @param config_path Path to the configuration file.
     * @param queue_size Size of the processing queue.
     * @param leaky Whether the queue is leaky.
     * @param print_fps Whether to print frames per second information.
     */
    PostprocessStage(std::string name, std::string so_path, std::string function_name=DEFAULT_FUNC_NAME, std::string config_path="", size_t queue_size=5, bool leaky=false, bool print_fps=false) : 
        ConnectedStage(name, queue_size, leaky, print_fps), m_so_path(so_path), m_config_path(config_path), m_function_name(function_name) {}

    /**
     * @brief Initialize the post-processing stage. by loading the provided so file with dlsym. 
     * If the binary has an init function to load parameters then it is called.
     * 
     * @return AppStatus Status of the initialization.
     */
    AppStatus init() override
    {
        // Load the give SO using dlopen.
        m_loaded_lib = dlopen(m_so_path.c_str(), RTLD_LAZY);
        if (!m_loaded_lib)
        {
            std::cerr << "Could not load lib " << dlerror() << std::endl;
            return AppStatus::CONFIGURATION_ERROR;
        }
        // Reset errors
        dlerror();

        // Expected .so can have an init function, get it if there is any
        auto init_func = (void *(*)(std::string, std::string))dlsym(m_loaded_lib, INIT_FUNC_NAME);
        if (init_func == nullptr)
        {
            // Set the library function handler with the requested function name
            m_handler_no_config = (void (*)(HailoROIPtr))dlsym(m_loaded_lib, m_function_name.c_str());
            m_params = nullptr;
        }
        else 
        {
            // Call the init function to get the params
            m_params = init_func(m_config_path, m_function_name);
            // Set the library function handler with the requested function name
            m_handler = (void (*)(HailoROIPtr, void *))dlsym(m_loaded_lib, m_function_name.c_str());
        }

        // If there was an error loading one of the symbols, close the dl and break.
        const char *dlsym_error = dlerror();
        if (dlsym_error)
        {
            std::cerr << "Cannot load symbol: " << dlsym_error << std::endl;
            dlclose(m_loaded_lib);
            return AppStatus::CONFIGURATION_ERROR;
        }

        m_last_time = std::chrono::steady_clock::now();

        return AppStatus::SUCCESS;
    }

    /**
     * @brief Deinitialize the post-processing stage loaded library.
     * 
     * @return AppStatus Status of the deinitialization.
     */
    AppStatus deinit() override
    {
        // Call the free function if there is any
        if (m_params != nullptr)
        {
            auto delete_func = (void (*)(void *))dlsym(m_loaded_lib, FREE_FUNC_NAME);
            if (delete_func != nullptr)
                delete_func(m_params);
        }
        // close the loaded library
        if (m_loaded_lib != nullptr)
        {
            dlclose(m_loaded_lib);
        }
    
        return AppStatus::SUCCESS;
    }

    /**
     * @brief Process the data in the buffer using the loaded library
     * 
     * @param data Buffer containing the data to be processed.
     * @return AppStatus Status of the processing.
     */
    AppStatus process(BufferPtr data)
    {    
        // Get the roi from the buffer
        HailoROIPtr hailo_roi = data->get_roi();

        // Call the handler with the roi and the params (if any)
        if (m_params != nullptr)
        {
            m_handler(hailo_roi, m_params);
        }
        else
        {
            m_handler_no_config(hailo_roi);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        if (m_print_fps)
        {
            std::cout << "Postprocess time (" << m_stage_name << ") = " << std::chrono::duration_cast<std::chrono::microseconds>(end - m_last_time).count() << "[microseconds]" << std::endl;
            m_last_time = end;
        }
        
	data->add_time_stamp(m_stage_name);
        set_duration(data);

	// Push the buffer to the next stage
        send_to_subscribers(data);
        return AppStatus::SUCCESS;
    }
};
