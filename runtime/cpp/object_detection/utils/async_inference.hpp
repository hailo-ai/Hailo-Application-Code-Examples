#ifndef _HAILO_ASYNC_INFERENCE_HPP_
#define _HAILO_ASYNC_INFERENCE_HPP_

#include "hailo/hailort.hpp"
#include "utils.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>

#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

using namespace hailort;

template<typename T>
class BoundedTSQueue {
private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond_not_empty;
    std::condition_variable m_cond_not_full;
    const size_t m_max_size;
    bool m_stopped;

public:
    explicit BoundedTSQueue(size_t max_size) : m_max_size(max_size), m_stopped(false) {}
    ~BoundedTSQueue() { stop(); }

    BoundedTSQueue(const BoundedTSQueue&) = delete;
    BoundedTSQueue& operator=(const BoundedTSQueue&) = delete;

    void push(T item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond_not_full.wait(lock, [this] { return m_queue.size() < m_max_size || m_stopped; });
        if (m_stopped) return;

        m_queue.push(std::move(item));
        m_cond_not_empty.notify_one();
    }

    bool pop(T &out_item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond_not_empty.wait(lock, [this] { return !m_queue.empty() || m_stopped; });
        if (m_stopped && m_queue.empty()) {
            return false;
        }

        out_item = std::move(m_queue.front());
        m_queue.pop();
        m_cond_not_full.notify_one();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopped = true;
        }
        m_cond_not_empty.notify_all();
        m_cond_not_full.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }
};


class AsyncModelInfer {
    private:
        std::unique_ptr<hailort::VDevice> vdevice;

        std::shared_ptr<hailort::InferModel> infer_model;
        hailort::ConfiguredInferModel configured_infer_model;
        hailort::ConfiguredInferModel::Bindings bindings;

        std::vector<std::shared_ptr<cv::Mat>> input_buffer_guards;
        std::vector<std::shared_ptr<uint8_t>> output_buffer_guards;
        std::map<std::string, hailo_vstream_info_t> output_vstream_info_by_name;
        std::shared_ptr<uint8_t> output_data_holder;
        std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> output_data_queue;

    public:
        // Constructors
        AsyncModelInfer() = default; // Default constructor
        AsyncModelInfer(std::shared_ptr<hailort::InferModel> infer_model);
        AsyncModelInfer(const std::string &hef_path,
                    std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue);

        AsyncModelInfer(const AsyncModelInfer&) = delete; // Copy constructor (deleted because of shared_ptr)
        AsyncModelInfer& operator=(const AsyncModelInfer&) = delete; // Copy assignment operator (deleted because of shared_ptr)
        AsyncModelInfer(AsyncModelInfer&& other) noexcept = default; // Move constructor
        AsyncModelInfer& operator=(AsyncModelInfer&& other) noexcept = default; // Move assignment
        ~AsyncModelInfer() = default; // Destructor

        // Getters
        const std::vector<hailort::InferModel::InferStream>& get_inputs();
        const std::vector<hailort::InferModel::InferStream>& get_outputs();
        const std::shared_ptr<hailort::InferModel> get_infer_model();
        std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> get_queue();

        // Functions
        void configure(std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> output_data_queue);
        void infer(std::shared_ptr<cv::Mat> input_data, cv::Mat original_frame);

        //Helpers
        void set_input_buffers(const std::shared_ptr<cv::Mat> &input_data);
        std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> prepare_output_buffers();
        void wait_and_run_async(cv::Mat original_frame,
                                const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos);
};

#endif /* _HAILO_ASYNC_INFERENCE_HPP_ */