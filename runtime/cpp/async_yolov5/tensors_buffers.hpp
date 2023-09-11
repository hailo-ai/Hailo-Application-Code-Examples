#ifndef _OUTPUT_TENSORS_HPP_
#define _OUTPUT_TENSORS_HPP_

#include "hailo/hailort.hpp"
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

constexpr int timeoutMs = 1000;

#if defined(__unix__)
#include <sys/mman.h>
#endif

using AlignedBuffer = std::shared_ptr<uint8_t>;
static AlignedBuffer page_aligned_alloc(size_t size)
{
#if defined(__unix__)
    auto addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return AlignedBuffer(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
    auto addr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!addr) throw std::bad_alloc();
    return AlignedBuffer(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}

template <typename T>
class ThreadSafeQueue {
public:
	ThreadSafeQueue() = default;
	// ThreadSafeQueue(const ThreadSafeQueue&) = delete;
	// ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
	// ThreadSafeQueue(ThreadSafeQueue&&) = delete;

	// void push(T& t) {
	// 	std::unique_lock<std::mutex> lock(m_mutex);
	// 	m_queue.push(t);
	// 	lock.unlock();
	// 	m_cond.notify_one();
	// }

	void push(T t) {
		std::unique_lock<std::mutex> lock(m_mutex);
		m_queue.push(t);
		lock.unlock();
		m_cond.notify_one();
	}

	T pop() {
		std::unique_lock<std::mutex> lock(m_mutex);
		auto timeout = std::chrono::milliseconds(timeoutMs);
		if (m_cond.wait_for(lock, timeout, [this]{ return !m_queue.empty(); })) {
			// m_cond.wait(lock, [this] { return !m_queue.empty(); });
			T t = m_queue.front();
			m_queue.pop();
			return t;
		}
		else {
			return nullptr;
		}
	}

	void pop(T& t) {
		std::unique_lock<std::mutex> lock(m_mutex);
		auto timeout = std::chrono::milliseconds(timeoutMs);
		if (m_cond.wait_for(lock, timeout, [this]{ return !m_queue.empty(); })) {
			// m_cond.wait(lock, [this] { return !m_queue.empty(); });
			t = m_queue.front();
			m_queue.pop();
		}
		else {
			t = nullptr;
		}
	}
private:
	std::queue<T> m_queue = {};
	std::mutex m_mutex = {};
	std::condition_variable m_cond ={};
};

class InputTensor {
public:
	InputTensor(hailo_stream_info_t info) : m_hw_data_bytes(info.hw_data_bytes), m_height(info.hw_shape.height), 
	m_width(info.hw_shape.width), m_channels(info.hw_shape.features) {}

	// InputTensor(uint32_t buffers_size, uint32_t height, uint32_t width, uint32_t channels) :
	// m_queue(buffers_size), m_height(height), m_width(width), m_channels(channels)
	// {}
	static bool sort_tensors_by_size (std::shared_ptr<InputTensor> i, std::shared_ptr<InputTensor> j) { 
		if(i->m_width == j->m_width){
			return i->m_channels < j->m_channels;
		}
		return i->m_width > j->m_width; 
	}

	ThreadSafeQueue<AlignedBuffer> m_queue;
	uint32_t m_hw_data_bytes;
	uint32_t m_height;
	uint32_t m_width;
	uint32_t m_channels;
};

class OutputTensor {
public:
	OutputTensor(hailo_stream_info_t info) : m_hw_data_bytes(info.hw_data_bytes), m_qp_zp(info.quant_info.qp_zp), 
	m_qp_scale(info.quant_info.qp_scale), m_height(info.hw_shape.height), m_width(info.hw_shape.width), m_channels(info.hw_shape.features) {} // TODO: not sure it's hw_data_bytes or hw_frame_size

	// OutputTensor(uint32_t buffers_size, float32_t qp_zp, float32_t qp_scale, uint32_t height, uint32_t width, uint32_t channels) :
    // m_queue(buffers_size), m_qp_zp(qp_zp), m_qp_scale(qp_scale), m_height(height), m_width(width), m_channels(channels) {}

    static bool sort_tensors_by_size (std::shared_ptr<OutputTensor> i, std::shared_ptr<OutputTensor> j) { 
		if(i->m_width == j->m_width){
			return i->m_channels < j->m_channels;
		}
		return i->m_width > j->m_width; 
	}

    ThreadSafeQueue<AlignedBuffer> m_queue;
	uint32_t m_hw_data_bytes;
    float32_t m_qp_zp;
    float32_t m_qp_scale;
	uint32_t m_height;
	uint32_t m_width;
	uint32_t m_channels;
};

class OutputTensors {
public:
	OutputTensors(int num_outputs) {
		outputs.reserve(num_outputs);
		for (size_t i = 0; i < 3; i++) {
			std::shared_ptr<OutputTensor> tensor(nullptr);
			outputs.emplace_back(tensor);
		}
	}

	OutputTensors() {
		OutputTensors(3);
	}

	OutputTensors(std::vector<hailo_stream_info_t> stream_infos) {
		outputs.reserve(stream_infos.size());
		for (auto& stream_info : stream_infos) {
			outputs.emplace_back(std::make_shared<OutputTensor>(stream_info));
		}
	}
	
	std::vector<std::shared_ptr<OutputTensor>> outputs;

};

#endif /* _OUTPUT_TENSORS_HPP_ */
