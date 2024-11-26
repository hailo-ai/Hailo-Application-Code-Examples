// general includes
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <tl/expected.hpp>
#include <signal.h>
#include <general/cxxopts.hpp>

// medialibrary includes
#include "media_library/encoder.hpp"
#include "media_library/frontend.hpp"
#include "media_library/signal_utils.hpp"

// infra includes
#include "infra/pipeline.hpp"
#include "infra/ai_stage.hpp"
#include "infra/dsp_stages.hpp"
#include "infra/postprocess_stage.hpp"
#include "infra/overlay_stage.hpp"
#include "infra/udp_stage.hpp"
#include "infra/tracker_stage.hpp"
#include "infra/aggregator_stage.hpp"
#include <zmq.hpp>

#define FRONTEND_CONFIG_FILE "/home/root/apps/ai_example_app/resources/configs/clip_frontend_config.json"
#define ENCODER_OSD_CONFIG_FILE(id) get_encoder_osd_config_file(id)
#define OUTPUT_FILE(id) get_output_file(id)

#define OVERLAY_STAGE "OverLay"
#define TRACKER_STAGE "Tracker"
#define UDP_0_STAGE "udp_0"
#define HOST_IP "10.0.0.2"

// AI Pipeline Params
#define AI_VISION_SINK "sink0" // The streamid from frontend to 4K stream that shows vision results 
#define AI_SINK "sink3" // The streamid from frontend to AI
// Detection AI Params
#define YOLO_HEF_FILE "/home/root/apps/ai_example_app/resources/yolov5s_personface_nv12.hef"
#define DETECTION_AI_STAGE "yolo_detection"
// Detection Postprocess Params
#define POST_STAGE "yolo_post"
#define YOLO_POST_SO "/usr/lib/hailo-post-processes/libyolo_hailortpp_post.so"
#define YOLO_FUNC_NAME "yolov5s_personface"
// Aggregator Params
#define AGGREGATOR_STAGE "aggregator"
#define AGGREGATOR_STAGE_2 "aggregator2"
// Callback Params
#define AI_CALLBACK_STAGE "ai_to_encoder"

// Tilling Params
#define TILLING_STAGE "tilling"
#define TILLING_INPUT_WIDTH 1920
#define TILLING_INPUT_HEIGHT 1080
#define TILLING_OUTPUT_WIDTH 640
#define TILLING_OUTPUT_HEIGHT 640
std::vector<HailoBBox> TILES = {{0.0,0.0,0.6,0.6},  {0.4,0,0.6,0.6},  
                                {0, 0.4, 0.6, 0.6},  {0.4, 0.4, 0.6, 0.6}, 
                                {0.0, 0.0, 1.0, 1.0}};

// Bbox crop Parms
#define BBOX_CROP_STAGE "bbox_crops"
#define BBOX_CROP_LABEL "person"
#define BBOX_CROP_INPUT_WIDTH 3840
#define BBOX_CROP_INPUT_HEIGHT 2160
#define BBOX_CROP_OUTPUT_WIDTH 224
#define BBOX_CROP_OUTPUT_HEIGHT 224

// CLIP AI Params
#define CLIP_HEF_FILE "/home/root/apps/ai_example_app/resources/clip_resnet_50_nv12.hef"
#define CLIP_AI_STAGE "clip"
// Clip Postprocess Params
#define CLIP_POST_STAGE "clip_post"
#define CLIP_POST_SO "/usr/lib/hailo-post-processes/libclip_post.so"
#define CLIP_FUNC_NAME "clip_resnet_50_nv12"

// Macro that turns coverts stream ids to port #s
#define PORT_FROM_ID(id) std::to_string(5000 + std::stoi(id.substr(4)) * 2)

enum class ArgumentType {
    Help,
    PrintFPS,
    PrintLatency,
    Timeout,
    Config,
    Error
};

/**
 * @brief Prints the help message for the application.
 *
 * This function displays the available command-line options and their descriptions.
 *
 * @param options The command-line options configuration.
 */
void print_help(const cxxopts::Options &options) {
    std::cout << options.help() << std::endl;
}

/**
 * @brief Builds the argument parser for the application.
 *
 * Defines the command-line arguments and their default values for the application.
 *
 * @return A cxxopts::Options object configured with the available arguments.
 */
cxxopts::Options build_arg_parser() {
    cxxopts::Options options("AI pipeline app");
    options.add_options()
        ("h,help", "Show this help")
        ("t,timeout", "Time to run", cxxopts::value<int>()->default_value("45"))
        ("f,print-fps", "Print FPS", cxxopts::value<bool>()->default_value("false"))
        ("l,print-latency", "Print Latency", cxxopts::value<bool>()->default_value("false"))
        ("c,config-file-path", "Frontend Configuration Path", cxxopts::value<std::string>()->default_value(FRONTEND_CONFIG_FILE));
    return options;
}

/**
 * @brief Handles parsed command-line arguments and converts them into argument types.
 *
 * This function processes the results of command-line argument parsing and maps
 * them to corresponding application-specific argument types.
 *
 * @param result The parsed command-line arguments.
 * @param options The original command-line options configuration.
 * @return A vector of ArgumentType enums representing the parsed arguments.
 */
std::vector<ArgumentType> handle_arguments(const cxxopts::ParseResult &result, const cxxopts::Options &options) {
    std::vector<ArgumentType> arguments;

    if (result.count("help")) {
        print_help(options);
        arguments.push_back(ArgumentType::Help);
    }

    if (result.count("print-fps")) {
        arguments.push_back(ArgumentType::PrintFPS);
    }

    if (result.count("timeout")) {
        arguments.push_back(ArgumentType::Timeout);
    }

    if (result.count("print-latency")) {
        arguments.push_back(ArgumentType::PrintLatency);
    }

    if (result.count("config-file-path")) {
        arguments.push_back(ArgumentType::Config);
    }

    // Handle unrecognized options
    for (const auto &unrecognized : result.unmatched()) {
        std::cerr << "Error: Unrecognized option or argument: " << unrecognized << std::endl;
        return {ArgumentType::Error};
    }

    return arguments;
}

/**
 * @brief Holds the resources required for the application.
 *
 * This structure contains pointers to various components and modules
 * used by the application, including the frontend, encoders, UDP outputs,
 * and the pipeline. It also includes a flag to control whether FPS (frames per second)
 * information should be printed.
 */
struct AppResources
{
    MediaLibraryFrontendPtr frontend;
    std::map<output_stream_id_t, MediaLibraryEncoderPtr> encoders;
    std::map<output_stream_id_t, UdpModulePtr> udp_outputs;
    PipelinePtr pipeline;
    bool print_fps;
    bool print_latency;
    std::string frontend_config;
};

/**
 * @brief Constructs the file path for the encoder OSD configuration.
 *
 * Generates the path to the encoder OSD configuration file for a given encoder ID.
 *
 * @param id The encoder ID.
 * @return A string representing the full path to the configuration file.
 */
inline std::string get_encoder_osd_config_file(const std::string &id)
{
    return "/home/root/apps/ai_example_app/resources/configs/encoder_osd_" + id + ".json";
}

/**
 * @brief Reads the contents of a file into a string.
 *
 * This function opens a file, reads its contents, and returns the data as a string.
 * Throws an exception if the file cannot be opened.
 *
 * @param file_path The path to the file to be read.
 * @return A string containing the file's contents.
 */
std::string read_string_from_file(const char *file_path)
{
    std::ifstream file_to_read;
    file_to_read.open(file_path);
    if (!file_to_read.is_open())
        throw std::runtime_error("config path is not valid");
    std::string file_string((std::istreambuf_iterator<char>(file_to_read)),
                            std::istreambuf_iterator<char>());
    file_to_read.close();
    std::cout << "Read config from file: " << file_path << std::endl;
    return file_string;
}

/**
 * @brief Deletes the contents of an output file.
 *
 * This function overwrites the specified output file with an empty binary file.
 * If the file cannot be written, an error message is printed.
 *
 * @param output_file The path to the output file to be cleared.
 */
void delete_output_file(std::string output_file)
{
    std::ofstream fp(output_file.c_str(), std::ios::out | std::ios::binary);
    if (!fp.good())
    {
        std::cout << "Error occurred at writing time!" << std::endl;
        return;
    }
    fp.close();
}
/**
 * @brief Subscribe elements within the application pipeline.
 *
 * This function subscribes the output streams from the frontend to appropriate
 * pipeline stages and encoders, ensuring that the data flows correctly through
 * the pipeline. It sets up callbacks for handling the data and integrates encoders
 * with UDP outputs.
 *
 * @param app_resources Shared pointer to the application's resources.
 */
void subscribe_elements(std::shared_ptr<AppResources> app_resources)
{
    // Get frontend output streams
    auto streams = app_resources->frontend->get_outputs_streams();
    if (!streams.has_value())
    {
        std::cout << "Failed to get stream ids" << std::endl;
        throw std::runtime_error("Failed to get stream ids");
    }

    // Subscribe to frontend
    FrontendCallbacksMap fe_callbacks;
    for (auto s : streams.value())
    {
        if (s.id == AI_SINK)
        {
            std::cout << "subscribing ai pipeline to frontend for '" << s.id << "'" << std::endl;
            app_resources->pipeline->get_stage_by_name(DETECTION_AI_STAGE)->add_queue(s.id);
            fe_callbacks[s.id] = [s, app_resources](HailoMediaLibraryBufferPtr buffer, size_t size)
            {
                BufferPtr wrapped_buffer = std::make_shared<Buffer>(buffer);
                app_resources->pipeline->get_stage_by_name(DETECTION_AI_STAGE)->push(wrapped_buffer, s.id);
            };
        }
        else if (s.id == AI_VISION_SINK)
        {
            std::cout << "subscribing to frontend for '" << s.id << "'" << std::endl;
            ConnectedStagePtr agg_stage = std::static_pointer_cast<ConnectedStage>(app_resources->pipeline->get_stage_by_name(AGGREGATOR_STAGE));
            fe_callbacks[s.id] = [s, app_resources, agg_stage](HailoMediaLibraryBufferPtr buffer, size_t size)
            {                      
                BufferPtr wrapped_buffer = std::make_shared<Buffer>(buffer);
                CroppingMetadataPtr cropping_meta = std::make_shared<CroppingMetadata>(1);
                wrapped_buffer->add_metadata(cropping_meta);
                agg_stage->push(wrapped_buffer, s.id);
            };
        }
        else
        {
            std::cout << "subscribing to frontend for '" << s.id << "'" << std::endl;
            fe_callbacks[s.id] = [s, app_resources](HailoMediaLibraryBufferPtr buffer, size_t size)
            {
                app_resources->encoders[s.id]->add_buffer(buffer);
            };
        }
    }
    app_resources->frontend->subscribe(fe_callbacks);

    // Subscribe to encoders
    for (const auto &entry : app_resources->encoders)
    {
        if (entry.first == AI_SINK)
        {
            // AI pipeline does not get an encoder since it is merged into 4K
            continue;
        }

        output_stream_id_t streamId = entry.first;
        MediaLibraryEncoderPtr encoder = entry.second;
        std::cout << "subscribing udp to encoder for '" << streamId << "'" << std::endl;
        app_resources->encoders[streamId]->subscribe(
            [app_resources, streamId](HailoMediaLibraryBufferPtr buffer, size_t size)
            {
                app_resources->udp_outputs[streamId]->add_buffer(buffer, size);
            });
    }

    // Subscribe ai stage to encoder
    std::cout << "subscribing ai pipeline to encoder '" << AI_VISION_SINK << "'" << std::endl;
    CallbackStagePtr ai_sink_stage = std::static_pointer_cast<CallbackStage>(app_resources->pipeline->get_stage_by_name(AI_CALLBACK_STAGE));
    ai_sink_stage->set_callback(
        [app_resources](BufferPtr data)
        {
            if (app_resources->print_latency) {
                app_resources->pipeline->print_latency();
            }
            app_resources->encoders[AI_VISION_SINK]->add_buffer(data->get_buffer());
        });
}

/**
 * @brief Create and configure an encoder and its corresponding UDP output file.
 *
 * This function sets up an encoder and a UDP output module for a given stream ID.
 * It reads configuration files and initializes the encoder and UDP module accordingly.
 *
 * @param id The ID of the output stream.
 * @param app_resources Shared pointer to the application's resources.
 */
void create_encoder_and_output_file(const std::string& id, std::shared_ptr<AppResources> app_resources)
{
    // Create and conifgure udp
    std::cout << "Creating encoder udp_" << id << std::endl;
    tl::expected<UdpModulePtr, AppStatus> udp_expected = UdpModule::create(id, HOST_IP, PORT_FROM_ID(id), EncodingType::H264);
    if (!udp_expected.has_value())
    {
        std::cout << "Failed to create udp" << std::endl;
        return;
    }
    app_resources->udp_outputs[id] = udp_expected.value();

    // Create and configure encoder
    std::cout << "Creating encoder enc_" << id << std::endl;
    std::string encoderosd_config_string = read_string_from_file(ENCODER_OSD_CONFIG_FILE(id).c_str());
    tl::expected<MediaLibraryEncoderPtr, media_library_return> encoder_expected = MediaLibraryEncoder::create(encoderosd_config_string, id);
    if (!encoder_expected.has_value())
    {
        std::cout << "Failed to create encoder osd" << std::endl;
        return;
    }
    app_resources->encoders[id] = encoder_expected.value();
}
/**
 * @brief Configure the frontend and encoders for the application.
 *
 * This function initializes the frontend and sets up encoders for each output stream
 * from the frontend. It reads configuration files to properly configure the components.
 *
 * @param app_resources Shared pointer to the application's resources.
 */
void configure_frontend_and_encoders(std::shared_ptr<AppResources> app_resources)
{
    // Create and configure frontend
    std::string frontend_config_string = read_string_from_file(app_resources->frontend_config.c_str());
    tl::expected<MediaLibraryFrontendPtr, media_library_return> frontend_expected = MediaLibraryFrontend::create(FRONTEND_SRC_ELEMENT_V4L2SRC, frontend_config_string);
    if (!frontend_expected.has_value())
    {
        std::cout << "Failed to create frontend" << std::endl;
        return;
    }
    app_resources->frontend = frontend_expected.value();

    // Get frontend output streams
    auto streams = app_resources->frontend->get_outputs_streams();
    if (!streams.has_value())
    {
        std::cout << "Failed to get stream ids" << std::endl;
        throw std::runtime_error("Failed to get stream ids");
    }

    // Create encoders and output files for each stream
    for (auto s : streams.value())
    {
        if (s.id == AI_SINK)
        {
            // AI pipeline does not get an encoder since it is merged into 4K
            continue;
        }

        create_encoder_and_output_file(s.id, app_resources);
    }
}

/**
 * @brief Start the application by initializing and starting all components.
 *
 * This function starts the UDP modules, encoders, pipeline stages, and the frontend
 * to begin processing data.
 *
 * @param app_resources Shared pointer to the application's resources.
 */
void start_app(std::shared_ptr<AppResources> app_resources)
{
    // Start each udp
    for (const auto &entry : app_resources->udp_outputs)
    {
        output_stream_id_t streamId = entry.first;
        UdpModulePtr udp = entry.second;

        std::cout << "starting UDP for " << streamId << std::endl;
        udp->start();
    }

    // Start each encoder
    for (const auto &entry : app_resources->encoders)
    {
        output_stream_id_t streamId = entry.first;
        MediaLibraryEncoderPtr encoder = entry.second;

        std::cout << "starting encoder for " << streamId << std::endl;
        encoder->start();
    }

    // Start stages pipeline
    app_resources->pipeline->start_pipeline();
    
    // Start frontend
    app_resources->frontend->start();
}
/**
 * @brief Stop the application by stopping all components.
 *
 * This function stops the frontend, pipeline stages, encoders, and UDP modules
 * to cease data processing and clean up resources.
 *
 * @param app_resources Shared pointer to the application's resources.
 */
void stop_app(std::shared_ptr<AppResources> app_resources)
{
    std::cout << "Stopping." << std::endl;
    // Stop frontend
    app_resources->frontend->stop();

    // Stop stages pipeline
    app_resources->pipeline->stop_pipeline();

    // Stop each encoder
    for (const auto &entry : app_resources->encoders)
    {
        entry.second->stop();
    }

    // Stop each udp
    for (const auto &entry : app_resources->udp_outputs)
    {
        entry.second->stop();
    }
}
/**
 * @brief Create and configure the application's processing pipeline.
 *
 * This function sets up the application's processing pipeline by creating various stages
 * and subscribing them to each other to form a complete pipeline. Each stage is initialized
 * with specific parameters and then added to the pipeline. The stages are also interconnected
 * by subscribing them to ensure data flows correctly between them.
 *
 * @param app_resources Shared pointer to the application's resources, which includes the pipeline object.
 */
void create_pipeline(std::shared_ptr<AppResources> app_resources)
{
    // Create pipeline
    app_resources->pipeline = std::make_shared<Pipeline>();

    // Create pipeline stages
    std::shared_ptr<HailortAsyncStage> detection_stage = std::make_shared<HailortAsyncStage>(DETECTION_AI_STAGE, YOLO_HEF_FILE, 5, 50 ,"device0", 10, 8, std::chrono::milliseconds(100), app_resources->print_fps);
    std::shared_ptr<PostprocessStage> detection_post_stage = std::make_shared<PostprocessStage>(POST_STAGE, YOLO_POST_SO, YOLO_FUNC_NAME, "", 5, false, app_resources->print_fps);
    std::shared_ptr<AggregatorStage> agg_stage = std::make_shared<AggregatorStage>(AGGREGATOR_STAGE, false, 
                                                                                   AI_VISION_SINK, 2, 
                                                                                   POST_STAGE, 10, 
                                                                                   true, 0.3, 0.1,
                                                                                   false, app_resources->print_fps);
    std::shared_ptr<TrackerStage> tracker_stage = std::make_shared<TrackerStage>(TRACKER_STAGE, 1, false, 1, app_resources->print_fps);
    std::shared_ptr<BBoxCropStage> bbox_crop_stage = std::make_shared<BBoxCropStage>(BBOX_CROP_STAGE, 100, BBOX_CROP_INPUT_WIDTH, BBOX_CROP_INPUT_HEIGHT,
                                                                                    BBOX_CROP_OUTPUT_WIDTH, BBOX_CROP_OUTPUT_HEIGHT,
                                                                                    AGGREGATOR_STAGE_2, CLIP_AI_STAGE, BBOX_CROP_LABEL, 1, false, app_resources->print_fps);
    std::shared_ptr<HailortAsyncStage> clip_stage = std::make_shared<HailortAsyncStage>(CLIP_AI_STAGE, CLIP_HEF_FILE, 20, 101 ,"device0", 1, 1, std::chrono::milliseconds(100), app_resources->print_fps);
    std::shared_ptr<PostprocessStage> clip_post_stage = std::make_shared<PostprocessStage>(CLIP_POST_STAGE, CLIP_POST_SO, CLIP_FUNC_NAME, "", 50, false, app_resources->print_fps);
    std::shared_ptr<AggregatorStage> agg_stage_2 = std::make_shared<AggregatorStage>(AGGREGATOR_STAGE_2, false, 
                                                                                     BBOX_CROP_STAGE, 2, 
                                                                                     CLIP_POST_STAGE, 30,
                                                                                     false, 0.3, 0.1,
                                                                                     false, app_resources->print_fps);
    std::shared_ptr<OverlayStage> overlay_stage = std::make_shared<OverlayStage>(OVERLAY_STAGE, 1, false, app_resources->print_fps);
    std::shared_ptr<CallbackStage> sink_stage = std::make_shared<CallbackStage>(AI_CALLBACK_STAGE, 1, false);
    
    // Add stages to pipeline
    app_resources->pipeline->add_stage(detection_stage);
    app_resources->pipeline->add_stage(detection_post_stage);
    app_resources->pipeline->add_stage(agg_stage);
    app_resources->pipeline->add_stage(tracker_stage);
    app_resources->pipeline->add_stage(bbox_crop_stage);
    app_resources->pipeline->add_stage(agg_stage_2);
    app_resources->pipeline->add_stage(overlay_stage);
    app_resources->pipeline->add_stage(sink_stage);
    app_resources->pipeline->add_stage(clip_stage);
    app_resources->pipeline->add_stage(clip_post_stage);

    // Subscribe stages to each other
    detection_stage->add_subscriber(detection_post_stage);
    detection_post_stage->add_subscriber(agg_stage);
    agg_stage->add_subscriber(tracker_stage);
    tracker_stage->add_subscriber(bbox_crop_stage);
    bbox_crop_stage->add_subscriber(agg_stage_2);
    bbox_crop_stage->add_subscriber(clip_stage);
    clip_stage->add_subscriber(clip_post_stage);
    clip_post_stage->add_subscriber(agg_stage_2);
    agg_stage_2->add_subscriber(overlay_stage);
    overlay_stage->add_subscriber(sink_stage);
}

/**
 * @brief Thread function to handle messages from a ZeroMQ subscriber.
 *
 * This function runs in an infinite loop, connecting to a ZeroMQ server, 
 * listening for messages, and processing them. If the "quit" message is received, 
 * it stops the application and terminates the thread.
 *
 * @param app_resources A shared pointer to the application's resources, used for 
 *                      stopping the app when the "quit" message is received.
 */
void zmq_subscriber_thread(std::shared_ptr<AppResources>  app_resources)
{
    //subscriber for quit message
    zmq::context_t context(1);
    zmq::socket_t subscriber(context, ZMQ_SUB);

    // Connect to the ZeroMQ server
    subscriber.connect("tcp://10.0.0.2:6000"); // Adjust the address and port as needed
    std::cout << "connected to 10.0.0.2:6000" << std::endl;
    subscriber.set(zmq::sockopt::subscribe, "");
    // Infinite loop to receive messages
    while (true) {
        // Receive the message
        zmq::message_t message;
        auto recv_result = subscriber.recv(message, zmq::recv_flags::none);
        if (!recv_result) {
            std::cerr << "Failed to receive message." << std::endl;
            continue;  // Skip processing if reception fails
        }
        // Convert the message to a string
        std::string received_message(static_cast<char*>(message.data()), message.size());

        // Print the received message
        std::cout << "Received message: " << received_message << std::endl;

        // Check if the message is "quit"
        if (received_message == "quit") {
            std::cout << "Quit message received. Exiting..." << std::endl;
            stop_app(app_resources);
            exit(0); 
        }
    }
}

/**
 * @brief Main function to initialize and run the application.
 *
 * This function sets up the application resources, registers a signal handler for SIGINT,
 * parses user arguments, configures the frontend and encoders, creates the pipeline,
 * subscribes elements, starts the pipeline, waits for a specified timeout, and then stops the pipeline.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return int Exit status of the application.
 */
int main(int argc, char *argv[])
{
    // App resources 
    std::shared_ptr<AppResources> app_resources = std::make_shared<AppResources>();
    app_resources->frontend_config = FRONTEND_CONFIG_FILE;

    // register signal SIGINT and signal handler
    signal_utils::register_signal_handler([app_resources](int signal)
    { 
        std::cout << "Stopping Pipeline..." << std::endl;
        // Stop pipeline
        stop_app(app_resources);
        // terminate program  
        exit(0); 
    });

    // Parse user arguments
    cxxopts::Options options = build_arg_parser();
    auto result = options.parse(argc, argv);
    std::vector<ArgumentType> argument_handling_results = handle_arguments(result, options);
    int timeout  = result["timeout"].as<int>();

    for (ArgumentType argument : argument_handling_results)
    {
        switch (argument)
        {
        case ArgumentType::Help:
            return 0;
        case ArgumentType::Timeout:
            break;
        case ArgumentType::PrintFPS:
            app_resources->print_fps = true;
            break;
        case ArgumentType::PrintLatency:
            app_resources->print_latency = true;
            break;
        case ArgumentType::Config:
            app_resources->frontend_config = result["config-file-path"].as<std::string>();
            break;
        case ArgumentType::Error:
            return 1;
        }
    }

    // Configure frontend and encoders
    configure_frontend_and_encoders(app_resources);

    // Create pipeline and stages
    create_pipeline(app_resources);

    // Subscribe elements
    subscribe_elements(app_resources);

    // Start pipeline
    start_app(app_resources);

    std::cout << "Using frontend config: " << app_resources->frontend_config << std::endl;

    std::thread zmq_thread(zmq_subscriber_thread, std::ref(app_resources));
    zmq_thread.detach();

    std::cout << "Started playing for " << timeout << " seconds." << std::endl;

    // Wait
    std::this_thread::sleep_for(std::chrono::minutes(timeout));
    // Stop pipeline
    stop_app(app_resources);

    return 0;
}