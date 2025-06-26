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
using Operation = std::function<void(const hailort::AsyncInferCompletionInfo&, cv::Mat org_frame)>;
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
        hailort::AsyncInferJob last_infer_job;
        std::map<std::string, hailo_vstream_info_t> output_vstream_info_by_name;

        //Helpers
        void set_input_buffers(const std::shared_ptr<cv::Mat> &input_data,
                               std::vector<std::shared_ptr<cv::Mat>> &input_guards);
        std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> prepare_output_buffers(
            std::vector<std::shared_ptr<uint8_t>> &output_guards);
        void wait_and_run_async(
            const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
            const std::vector<std::shared_ptr<uint8_t>> &output_guards,
            const std::vector<std::shared_ptr<cv::Mat>> &input_guards,
            std::function<void(const hailort::AsyncInferCompletionInfo&,
                const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                const std::vector<std::shared_ptr<uint8_t>> &)> callback);
    public:
        // Constructors

        // Constructor for when using only one model one device
        AsyncModelInfer(const std::string &hef_path);
        // Constructor for when using multiple models on the same device
        AsyncModelInfer(const std::string &hef_path, const std::string &group_id);
        // Destructor
        ~AsyncModelInfer();

        // Getters
        const std::vector<hailort::InferModel::InferStream>& get_inputs();
        const std::vector<hailort::InferModel::InferStream>& get_outputs();
        const std::shared_ptr<hailort::InferModel> get_infer_model();
        
        // Functions
        void infer(
            std::shared_ptr<cv::Mat> input_data,
            std::function<void(
                const hailort::AsyncInferCompletionInfo &,
                const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                const std::vector<std::shared_ptr<uint8_t>> &)> callback);

};
#endif /* _HAILO_ASYNC_INFERENCE_HPP_ */