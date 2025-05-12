#pragma once

// general includes
#include <thread>
#include <vector>

// medialibrary includes
#include "hailo/media_library/buffer_pool.hpp"

// tappas includes
#include "hailo_objects.hpp"

class Buffer;
using BufferPtr = std::shared_ptr<Buffer>;

enum class MetadataType
{
    UNKNOWN,
    TENSOR,
    EXPECTED_CROPS,
};

class Metadata 
{
private:
    MetadataType m_type;
public:
    Metadata(MetadataType type=MetadataType::UNKNOWN) : m_type(type)
    {}

    virtual ~Metadata() = default;

    MetadataType get_type()
    {
        return m_type;
    }
};
using MetadataPtr = std::shared_ptr<Metadata>;

class CroppingMetadata : public Metadata 
{
private:
    int m_num_crops;
public:
    CroppingMetadata(int num_crops) : Metadata(MetadataType::EXPECTED_CROPS), m_num_crops(num_crops)
    {}

    int get_num_crops()
    {
        return m_num_crops;
    }
};
using CroppingMetadataPtr = std::shared_ptr<CroppingMetadata>;

class BufferMetadata : public Metadata
{
private:
    BufferPtr m_buffer;
public:
    BufferMetadata(BufferPtr buffer, MetadataType type=MetadataType::UNKNOWN) : Metadata(type), m_buffer(buffer)
    {}

    BufferPtr get_buffer()
    {
        return m_buffer;
    }
};
using BufferMetadataPtr = std::shared_ptr<BufferMetadata>;

class TensorMetadata : public BufferMetadata
{
private:
    std::string m_tensor_name;
public:
    TensorMetadata(BufferPtr buffer, std::string tensor_name) : BufferMetadata(buffer, MetadataType::TENSOR), m_tensor_name(tensor_name)
    {}

    std::string get_tensor_name()
    {
        return m_tensor_name;
    }
};
using TensorMetadataPtr = std::shared_ptr<TensorMetadata>;

class TimeStamp
{
private:
    std::chrono::steady_clock::time_point m_time;
    std::string m_stage_name;

public:
    TimeStamp(std::string stage_name)
    {
        m_time = std::chrono::steady_clock::now();
        m_stage_name = stage_name;
    }

    std::chrono::steady_clock::time_point get_time_point() const {
        return m_time;
    }

    std::string get_stage_name() const {
        return m_stage_name;
    }
};
using TimeStampPtr = std::shared_ptr<TimeStamp>;



class Buffer {
private:
    HailoMediaLibraryBufferPtr m_buffer;
    HailoROIPtr m_roi;
    std::vector<MetadataPtr> m_metadata;
    std::vector<TimeStampPtr> m_timestamps;

public:
    Buffer(HailoMediaLibraryBufferPtr buffer)
        : m_buffer(buffer) 
    {
        m_roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        TimeStampPtr time_stamp =  std::make_shared<TimeStamp>("Source");
        m_timestamps.push_back(time_stamp);
    }


    Buffer(HailoMediaLibraryBufferPtr buffer, HailoROIPtr roi)
        : m_buffer(buffer) 
    {
        if (roi) {
            m_roi = roi;
        } else {
            m_roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        }
    }

    HailoMediaLibraryBufferPtr get_buffer() const {
        return m_buffer;
    }

    HailoROIPtr get_roi() const {
        return m_roi;
    }

    void add_time_stamp(const std::string &stage) {
        TimeStampPtr time_stamp =  std::make_shared<TimeStamp>(stage);
        m_timestamps.push_back(time_stamp);
    }

    size_t get_num_stages() const {
            return m_timestamps.size();
    }

    std::chrono::steady_clock::time_point get_time_stamp( int index) const {
        return  m_timestamps[index]->get_time_point();
    }

    void print_latency_measurements() const {
         size_t last_index = (m_timestamps.size() - 1);
          for (size_t i = 0; i < m_timestamps.size(); ++i){
            std::cout  << m_timestamps[i]->get_stage_name()<< " ";
          }
    
          std::cout << std::endl;
          for (size_t i = 0; i < last_index; ++i) {
            auto stage_name = m_timestamps[i+1]->get_stage_name();
            std::chrono::steady_clock::time_point end  = m_timestamps[i+1]->get_time_point();
            std::chrono::steady_clock::time_point start  = m_timestamps[i]->get_time_point();
            
            std::cout  << stage_name <<" took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[milli] ";
          }
          std::cout  <<  std::endl;
    }
        
    void add_metadata(MetadataPtr metadata) {
        m_metadata.push_back(metadata);
    }

    void remove_metadata(MetadataPtr metadata) {
        for (auto it = m_metadata.begin(); it != m_metadata.end(); ++it) {
            if (*it == metadata) {
                m_metadata.erase(it);
                break;
            }
        }
    }

    std::vector<MetadataPtr> get_metadata_of_type(MetadataType metadata_type) {
        std::vector<MetadataPtr> metadata;
        for (auto &m : m_metadata) {
            if (m->get_type() == metadata_type) {
                metadata.push_back(m);
            }
        }
        return metadata;
    }

};
