/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

#include "clip.hpp"
#include "zmq.hpp"
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <hailo_common.hpp>


const char *output_layer_name = "clip_resnet_50/conv59";

zmq::context_t zmq_context;
zmq::socket_t zmq_publisher;
zmq::socket_t zmq_subscriber;

bool new_prompt = false;
bool initialization_done = false; 
bool first_running = true;

std::queue<std::vector<std::vector<float>>> m_text_embedding_queue;
std::vector<std::string> prompts;
std::vector<std::vector<float>> text_embeddings;
std::vector<bool> negatives;
float threshold = 0.0;

std::vector<float> probs;


float logit_scale_1 = std::exp(4.60517); 


std::mutex image_queue_mutex;  
std::mutex text_queue_mutex;   
std::mutex new_prompt_mutex; 
std::mutex first_running_mutex; 
std::mutex initialization_mutex; 

using json = nlohmann::json;

/**
 * @brief Initialize the ZeroMQ PUB socket for publishing messages.
 * 
 * @param address The address to bind the PUB socket to.
 */
void init_zmq_publisher(const std::string &address) {
    try {
        zmq_publisher = zmq::socket_t(zmq_context, zmq::socket_type::pub);
        zmq_publisher.bind(address);
    } catch (const zmq::error_t &e) {
        std::cerr << "Error initializing ZMQ publisher: " << e.what() << std::endl;
    }
}

/**
 * @brief Initialize the ZeroMQ SUB socket for subscribing to messages.
 * 
 * @param address The address to connect the SUB socket to.
 */
void init_zmq_subscriber(const std::string &address) {
    zmq_subscriber = zmq::socket_t(zmq_context, ZMQ_SUB);
    zmq_subscriber.connect(address);
    zmq_subscriber.set(zmq::sockopt::subscribe, ""); 
}

/**
 * @brief Receive and process messages from the publisher using ZeroMQ SUB socket.
 * 
 * @param subscriber The ZeroMQ subscriber socket.
 */
void receive_messages(zmq::socket_t &zmq_subscriber) {
    try {
        while (true) {
            zmq::message_t zmq_msg;
            (void) zmq_subscriber.recv(zmq_msg, zmq::recv_flags::none);
            std::string message_str(static_cast<char*>(zmq_msg.data()), zmq_msg.size());
            json received_json = json::parse(message_str);

            if (received_json.contains("prompts") && received_json["prompts"].is_array()) {
                prompts.clear(); 
                for (const auto& item : received_json["prompts"]) {
                    if (item.is_null()) {
                        prompts.push_back(""); 
                    } else {
                        prompts.push_back(item.get<std::string>());
                    }
                }
            }

            if (received_json.contains("embedding") && received_json["embedding"].is_array()) {
                text_embeddings.clear(); 
                for (const auto& item : received_json["embedding"]) {
                    if (item.is_null()) {
                        text_embeddings.push_back({}); 
                    } else {
                        text_embeddings.push_back(item.get<std::vector<float>>());
                    }
                }
            }

            if (received_json.contains("negatives") && received_json["negatives"].is_array()) {
                negatives.clear(); 
                for (const auto& item : received_json["negatives"]) {
                    if (item.is_null()) {
                        negatives.push_back({});
                    } else {
                        negatives.push_back(item.get<bool>());
                    }
                }
            }

            if (received_json.contains("threshold")) {
                threshold = received_json["threshold"].get<float>();
            }

            std::lock_guard<std::mutex> lock(text_queue_mutex);
            m_text_embedding_queue.push(text_embeddings);
        }
    } catch (const zmq::error_t& e) { }
}

/**
 * @brief Send results using ZeroMQ PUB socket.
 * 
 * @param result A pair of integers representing the result.
 */
void send_results(const std::pair<int,int> result) {
    size_t size = 2 * sizeof(int);
    zmq::message_t message(size);
    
    int* data = static_cast<int*>(message.data());
    data[0] = result.first;
    data[1] = result.second;

    zmq_publisher.send(message, zmq::send_flags::none);
}

/**
 * @brief Apply the softmax function to a vector of logits.
 * 
 * @param logits A vector of logits.
 * @return A vector of probabilities.
 */
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exp_logits(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    float sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] == 0.0f) {
            continue;
        }
        exp_logits[i] = std::exp(logits[i] - max_logit); 
        sum_exp += exp_logits[i];
    }
    
    for (size_t i = 0; i < exp_logits.size(); ++i) {
        exp_logits[i] /= sum_exp;
    }
    
    return exp_logits;
}

/**
 * @brief Normalize a vector.
 * 
 * @param vec A vector to be normalized.
 */
void normalize(std::vector<float>& vec) {
    float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));

    if (norm != 0.0f) {
        std::transform(vec.begin(), vec.end(), vec.begin(), [norm](float v) { return v / norm; });
    }
}

/**
 * @brief Normalize a 2D vector.
 * 
 * @param data A 2D vector to be normalized.
 */
void normalize_vectors(std::vector<std::vector<float>>& data) {
    for (auto& vec : data) {
        float norm = std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0f, [](float sum, float val) {
            return sum + val * val;
        }));

        if (norm == 0.0f) {
            continue;
        }

        for (auto& val : vec) {
            val /= norm;
        }
    }
}

/**
 * @brief Compute the dot product between a vector and a 2D vector.
 * 
 * @param A A vector.
 * @param B A 2D vector.
 * @return A vector of dot product results.
 */
std::vector<float> custom_dot_product(const std::vector<float>& A, const std::vector<std::vector<float>>& B) {
    std::vector<float> result(B.size());

    for (std::size_t i = 0; i < B.size(); ++i) {
        if(B[i].size() == 0){
            result[i] = 0.0f;
        }
        else{
            result[i] = std::inner_product(A.begin(), A.end(), B[i].begin(), 0.0f);
            result[i] = result[i] * logit_scale_1;
        }
    }

    return result;
}

/**
 * @brief Calculate probabilities and send them.
 *
 * This function normalizes the image embeddings and text embeddings, 
 * computes the dot product between them, and then applies the softmax 
 * function to get the probabilities.
 *
 * @param text_embeddings A 2D vector containing text embeddings.
 * @param image_embeddings A vector containing image embeddings.
 */
void calc_and_send_probs(std::vector<std::vector<float>>& text_embeddings, std::vector<float>& image_embeddings) {
    normalize(image_embeddings);
    normalize_vectors(text_embeddings); 

    std::vector<float> dot_product_result = custom_dot_product(image_embeddings, text_embeddings);

    probs = softmax(dot_product_result);
}

/**
 * @brief Get the image embedding and push it to the image embedding queue.
 *
 * This function retrieves the tensor from the given ROI, dequantizes the 
 * tensor data, and returns it as a vector of floats.
 *
 * @param roi A pointer to the region of interest (ROI).
 * @return A vector of floats representing the dequantized image embedding.
 */
std::vector<float> get_image_embedding(HailoROIPtr roi){
    HailoTensorPtr tensor = roi->get_tensor(output_layer_name);
    if(tensor) {
        std::unique_lock<std::mutex> lock(image_queue_mutex);
        uint8_t *data_ptr = tensor->data();
        size_t data_size = tensor->size();

        std::vector<float> dequantized_data(data_size);
        // Dequantize the tensor data
        for (size_t i = 0; i < data_size; ++i) {
            dequantized_data[i] = tensor->fix_scale(data_ptr[i]);
        }
        return dequantized_data;
    }

    return {};
}

/**
 * @brief Get the unique tracking ID from a detection.
 *
 * This function retrieves the unique tracking ID from the given detection.
 *
 * @param detection A pointer to the detection object.
 * @return A pointer to the unique tracking ID.
 */
HailoUniqueIDPtr get_tracking_id(HailoDetectionPtr detection)
{
    for (auto obj : detection->get_objects_typed(HAILO_UNIQUE_ID))
    {
        HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
        if (id->get_mode() == TRACKING_ID)
        {
            return id;
        }
    }
    return nullptr;
}

/**
 * @brief Process the ROI using the CLIP model.
 * 
 * @param roi A pointer to the region of interest (ROI).
 */
void clip(HailoROIPtr roi){
    std::lock_guard<std::mutex> lock(initialization_mutex);

    // If this is the first run, we need to initialize ZMQ and start the threads.
    if(!initialization_done){
        init_zmq_publisher("tcp://10.0.0.1:7000");
        init_zmq_subscriber("tcp://10.0.0.2:5555");

        std::thread subscriber_thread(receive_messages, std::ref(zmq_subscriber)); 

        subscriber_thread.detach();
        initialization_done = true;
    }

    auto image_embedding = get_image_embedding(roi);
    calc_and_send_probs(text_embeddings, image_embedding);

    std::shared_ptr<HailoDetection> detection = std::dynamic_pointer_cast<HailoDetection>(roi);

    if(detection){
        int tracking_id = get_tracking_id(detection)->get_id();
        hailo_common::remove_classifications(roi, "clip");

        if(prompts.size() > 0){
            auto max_prob = std::max_element(probs.begin(), probs.end());
            int index = std::distance(probs.begin(), max_prob);
            std::string label = prompts[index];
            const std::string prefix = "A photo of ";
            if (label.rfind(prefix, 0) == 0) { 
                label.erase(0, prefix.length()); 
            }
            if(!negatives[index] && *max_prob > threshold && label != ""){
                hailo_common::add_classification(roi,
                                    std::string("clip"),
                                    label,
                                    *max_prob,
                                    -1);
                send_results(std::pair<int,int>(tracking_id, index));
            }
            else{
                send_results(std::pair<int,int>(tracking_id, -1));
            }
        }
    }
}

/**
 * @brief Process the ROI using the CLIP ResNet-50 model with NV12 format.
 * 
 * @param roi A pointer to the region of interest (ROI).
 */
void clip_resnet_50_nv12(HailoROIPtr roi) {
    output_layer_name = "clip_resnet_50/conv59";
    clip(roi);
}