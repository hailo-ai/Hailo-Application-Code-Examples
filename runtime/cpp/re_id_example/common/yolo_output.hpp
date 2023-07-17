
#ifndef _HAILO_YOLOV5_OL_HPP_
#define _HAILO_YOLOV5_OL_HPP_
#include "hailo_objects.hpp"

/**
 * @brief Base class to represent OutputLayer of Yolo networks.
 *
 */
class YoloOutputLayer
{
public:
    static const uint NUM_ANCHORS = 3;
    YoloOutputLayer(uint width, uint height, uint num_of_classes, std::vector<int> anchors, bool perform_sigmoid, int label_offset)
        : _width(width), _height(height), _num_classes(num_of_classes), _anchors(anchors), label_offset(label_offset), _perform_sigmoid(perform_sigmoid){};
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
    virtual float get_confidence(uint row, uint col, uint anchor) = 0;
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
    float sigmoid(float x);
    /**
     * @brief Get the class channel object
     *
     * @param anchor
     * @param channel
     * @return uint
     */
    virtual uint get_class_prob(uint row, uint col, uint anchor, uint class_id) = 0;
    /**
     * @brief Get the class conf object
     *
     * @param prob_max
     * @return float
     */
    virtual float get_class_conf(uint prob_max) = 0;
};

class YoloSplittedOutputLayer : public YoloOutputLayer
{
public:
    YoloSplittedOutputLayer(HailoTensorPtr center,
                            HailoTensorPtr scale,
                            HailoTensorPtr obj,
                            HailoTensorPtr cls,
                            std::vector<int> anchors,
                            bool perform_sigmoid,
                            int label_offset)
        : YoloOutputLayer(cls->width(), cls->height(), (uint)(cls->features() / NUM_ANCHORS), anchors, perform_sigmoid, label_offset),
          _center(center), _scale(scale), _obj(obj), _cls(cls){};
    virtual float get_confidence(uint row, uint col, uint anchor);
    virtual uint get_class_prob(uint row, uint col, uint anchor, uint channel);
    virtual float get_class_conf(uint prob_max);
    virtual std::pair<float, float> get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height);

protected:
    HailoTensorPtr _center;
    HailoTensorPtr _scale;
    HailoTensorPtr _obj;
    HailoTensorPtr _cls;
};

class Yolov3OL : public YoloSplittedOutputLayer
{
public:
    Yolov3OL(HailoTensorPtr center, HailoTensorPtr scale,
             HailoTensorPtr obj, HailoTensorPtr cls, std::vector<int> anchors, int label_offset)
        : YoloSplittedOutputLayer(center, scale, obj, cls, anchors, true, label_offset){};
    virtual std::pair<float, float> get_center(uint row, uint col, uint anchor);
};

class Yolov4OL : public YoloSplittedOutputLayer
{
public:
    const float SCALE_XY = 1.05f;
    Yolov4OL(HailoTensorPtr center,
             HailoTensorPtr scale,
             HailoTensorPtr obj,
             HailoTensorPtr cls,
             std::vector<int> anchors, int label_offset) : YoloSplittedOutputLayer(center, scale, obj, cls, anchors, false, label_offset){};
    virtual std::pair<float, float> get_center(uint row, uint col, uint anchor);
};

class Yolov5OL : public YoloOutputLayer
{
public:
    static const uint NUM_CENTERS = 2;
    static const uint NUM_SCALES = 2;
    static const uint NUM_CONF = 1;
    static const uint CONF_CHANNEL_OFFSET = NUM_CENTERS + NUM_SCALES;
    static const uint CLASS_CHANNEL_OFFSET = CONF_CHANNEL_OFFSET + NUM_CONF;
    Yolov5OL(HailoTensorPtr tensor,
             std::vector<int> anchors, bool perform_sigmoid, int label_offset)
        : YoloOutputLayer(tensor->width(), tensor->height(), num_classes(tensor->features()), anchors, false, label_offset), _tensor(tensor)
    {
        _anchor_size = _tensor->features() / NUM_ANCHORS;
    };

    /**
     * @brief return the output layer's number of classes.
     *
     * @param channels the number of features of the tensor.
     * @return uint
     */
    static uint num_classes(uint channels)
    {
        return (channels / NUM_ANCHORS) - CLASS_CHANNEL_OFFSET;
    }

    virtual float get_confidence(uint row, uint col, uint anchor);
    virtual float get_class_conf(uint prob_max);
    virtual uint get_class_prob(uint row, uint col, uint anchor, uint channel);
    virtual std::pair<float, float> get_center(uint row, uint col, uint anchor);
    virtual std::pair<float, float> get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height);

private:
    HailoTensorPtr _tensor;
    uint _anchor_size;
};

class YoloXOL : public YoloOutputLayer
{
public:
    static const uint NUM_ANCHORS = 1;
    YoloXOL(HailoTensorPtr bbox,
            HailoTensorPtr obj,
            HailoTensorPtr cls,
            int label_offset)
        : YoloOutputLayer(cls->width(), cls->height(), (uint)(cls->features()), {}, false, label_offset),
          _bbox(bbox), _obj(obj), _cls(cls){};
    virtual float get_confidence(uint row, uint col, uint anchor);
    virtual uint get_class_prob(uint row, uint col, uint anchor, uint channel);
    virtual float get_class_conf(uint prob_max);
    virtual std::pair<float, float> get_center(uint row, uint col, uint anchor);
    virtual std::pair<float, float> get_shape(uint row, uint col, uint anchor, uint image_width, uint image_height);

protected:
    HailoTensorPtr _bbox;
    HailoTensorPtr _obj;
    HailoTensorPtr _cls;
};

#endif