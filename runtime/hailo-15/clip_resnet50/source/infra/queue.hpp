#pragma once

// General includes
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

// Infra includes
#include "buffer.hpp"

class Queue
{
private:
    std::queue<BufferPtr> m_queue;
    size_t m_max_buffers;
    bool m_leaky;
    bool m_print_level;
    std::string m_name;
    bool m_flushing;
    std::unique_ptr<std::condition_variable> m_condvar;
    std::shared_ptr<std::mutex> m_mutex;
    uint64_t m_drop_count = 0, m_push_count = 0;

public:
    Queue(std::string name, size_t max_buffers, bool leaky=false, bool print_level=false)
        : m_max_buffers(max_buffers), m_leaky(leaky), m_print_level(print_level), m_name(name), m_flushing(false)
    {
        m_mutex = std::make_shared<std::mutex>();
        m_condvar = std::make_unique<std::condition_variable>();
        m_queue = std::queue<BufferPtr>();
    }

    ~Queue()
    {
        m_flushing = true;
        m_condvar->notify_all();
        flush();
    }

    std::string name()
    {
        return m_name;
    }

    int size()
    {
        std::unique_lock<std::mutex> lock(*(m_mutex));
        return m_queue.size();
    }

    void push(BufferPtr buffer)
    {
        std::unique_lock<std::mutex> lock(*(m_mutex));
        if (!m_leaky)
        {
            // if not leaky, then wait until there is space in the queue
            m_condvar->wait(lock, [this]
                            { return m_queue.size() < m_max_buffers; });
        } 
        else 
        {
            // if leaky, pop the front for a full queue
            if(m_queue.size() >= m_max_buffers)
            {
                m_queue.pop();
                m_drop_count++;
            }
        }
        m_queue.push(buffer);
        m_push_count++;
        if (m_print_level)
        {
            std::cout << "Queue: " << m_name << " level: " << m_queue.size() << std::endl;
        }
        m_condvar->notify_one();
    }

    BufferPtr pop()
    {
        std::unique_lock<std::mutex> lock(*(m_mutex));
        // wait for there to be something in the queue to pull
        m_condvar->wait(lock, [this]
                            { return !m_queue.empty() || m_flushing; });
        if (m_queue.empty())
        {
            // if we reachied here, then the queue is empty and we are flushing
            return nullptr;
        }
        BufferPtr buffer = m_queue.front();
        m_queue.pop();
        m_condvar->notify_one();
        return buffer;
    }

    void flush()
    {
        std::unique_lock<std::mutex> lock(*(m_mutex));
        m_flushing = true;
        while (!m_queue.empty())
        {
            m_queue.pop();
        }
        m_condvar->notify_all();
    }

};
using QueuePtr = std::shared_ptr<Queue>;