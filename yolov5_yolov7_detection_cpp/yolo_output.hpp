/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once
#include "common/hailo_objects.hpp"
#include <iostream>

/**
 * @brief Base class to represent OutputLayer of Yolo networks.
 *
 */
class YoloOutputLayer
{
public:
    static const uint NUM_ANCHORS = 3;
    static const uint NUM_CENTERS = 2;
    static const uint NUM_SCALES = 2;
    static const uint NUM_CONF = 1;
    static const uint CONF_CHANNEL_OFFSET = NUM_CENTERS + NUM_SCALES;
    static const uint CLASS_CHANNEL_OFFSET = CONF_CHANNEL_OFFSET + NUM_CONF;
    YoloOutputLayer(uint width,
                    uint height,
                    uint num_of_classes,
                    std::vector<int> anchors,
                    bool perform_sigmoid,
                    int label_offset,
                    bool is_uint16,
                    HailoTensorPtr tensor = nullptr) : _width(width),
                                                       _height(height),
                                                       _num_classes(num_of_classes),
                                                       _anchors(anchors),
                                                       label_offset(label_offset),
                                                       _perform_sigmoid(perform_sigmoid),
                                                       _is_uint16(is_uint16),
                                                       _tensor(tensor){};
    virtual ~YoloOutputLayer() = default;

    uint _width;
    uint _height;
    uint _num_classes;
    std::vector<int> _anchors;
    int label_offset;

    /**
     * @brief Get the class object
     *
     * @param row
     * @param col
     * @param anchor
     * @return std::pair<uint, float> class id and class probability.
     */
    std::pair<uint, float> get_class(uint row, uint col, uint anchor);
    /**
     * @brief Get the confidence object
     *
     * @param row
     * @param col
     * @param anchor
     * @return float
     */
    virtual float get_confidence(uint row, uint col, uint anchor);
    /**
     * @brief Get the center object
     *
     * @param row
     * @param col
     * @param anchor
     * @return std::pair<float, float> pair of x,y of the center of this prediction.
     */
    virtual std::pair<float, float> get_center(uint row, uint col, uint anchor) = 0;
    /**
     * @brief Get the shape object
     *
     * @param row
     * @param col
     * @param anchor
     * @param image_width
     * @param image_height
     * @return std::pair<float, float> pair of w,h of the shape of this prediction.
     */
    virtual std::pair<float, float> get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height) = 0;

protected:
    bool _perform_sigmoid;
    bool _is_uint16;
    HailoTensorPtr _tensor;
    float sigmoid(float x);
    /**
     * @brief Get the class channel object
     *
     * @param anchor
     * @param channel
     * @return uint
     */
    virtual uint get_class_prob(uint row, uint col, uint anchor, uint class_id);
    /**
     * @brief Get the class conf object
     *
     * @param prob_max
     * @return float
     */
    virtual float get_class_conf(uint prob_max) = 0;
    static uint num_classes(uint channels)
    {
        return (channels / NUM_ANCHORS) - CLASS_CHANNEL_OFFSET;
    }
};

class Yolov5OL : public YoloOutputLayer
{
public:
    Yolov5OL(HailoTensorPtr tensor,
             std::vector<int> anchors,
             bool perform_sigmoid,
             int label_offset,
             bool is_uint16) : YoloOutputLayer(tensor->width(),
                                               tensor->height(),
                                               num_classes(tensor->features()),
                                               anchors,
                                               perform_sigmoid,
                                               label_offset,
                                               is_uint16,
                                               tensor){};
    virtual float get_class_conf(uint prob_max);
    virtual std::pair<float, float> get_center(uint row, uint col, uint anchor);
    virtual std::pair<float, float> get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height);
};