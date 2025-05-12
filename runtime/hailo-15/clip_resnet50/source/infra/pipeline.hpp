#pragma once

// general includes
#include <queue>
#include <thread>
#include <vector>

// infra includes
#include "stage.hpp"

class Pipeline
{
private:
    std::vector<StagePtr> m_stages;

public:

    void add_stage(StagePtr stage)
    {
        m_stages.push_back(stage);
    }

    void start_pipeline()
    {
        for (auto &stage : m_stages)
        {
            stage->start();
        }
    }

    void stop_pipeline()
    {
        for (auto &stage : m_stages)
        {
            stage->stop();
        }
    }

    StagePtr get_stage_by_name(std::string stage_name)
    {
        for (auto &stage : m_stages)
        {
            if (stage->get_name() == stage_name)
            {
                return stage;
            }
        }
        return nullptr;
    }

    void print_latency()
    {
        std::cout << "Stage latencies ";
        for (auto &stage : m_stages)
        {
            std::cout << stage->get_name() << " : " << stage->get_duration().count() << "  ";
        }
        std::cout << std::endl;
    }
};
using PipelinePtr = std::shared_ptr<Pipeline>;