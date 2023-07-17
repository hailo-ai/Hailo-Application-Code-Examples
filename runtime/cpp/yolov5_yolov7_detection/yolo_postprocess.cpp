/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <typeinfo>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>

#include "yolo_postprocess.hpp"
#include "common/nms.hpp"
#include "json_config.hpp"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

class YoloPost
{
protected:
    std::vector<std::shared_ptr<YoloOutputLayer>> _layers;
    uint _max_boxes;
    float _detection_thr;
    float _iou_thr;
    uint m_image_width;
    uint m_image_height;
    std::map<uint8_t, std::string> m_dataset;

public:
    virtual ~YoloPost() = default;
    YoloPost(std::map<uint8_t, std::string> dataset,
             float detection_threshold,
             float iou_threshold,
             uint max_boxes)
        : _max_boxes(max_boxes), _detection_thr(detection_threshold),
          _iou_thr(iou_threshold), m_dataset(dataset){};

    std::vector<HailoDetection> decode()
    {
        std::vector<HailoDetection> objects;
        objects.reserve(_max_boxes);
        for (auto layer : _layers)
        {
            extract_boxes(layer, objects);
        }
        common::nms(objects, _iou_thr);
        if (objects.size() > _max_boxes)
        {
            HailoBBox bbox(0, 0, 1, 1);
            HailoDetection empty_detection(bbox, "None", 0.0);
            objects.resize(_max_boxes, empty_detection);
        }

        return objects;
    }

    uint get_num_classes()
    {
        return _layers[0]->_num_classes;
    }

    /**
     * @brief Extract the boxes of generic yolo output layer.
     *
     * @param[in] image_size Network's input image width/height.
     * @param[in] thr Postprocess threshold.
     * @param[out] objects Reference to vector of detections.
     */
    void extract_boxes(std::shared_ptr<YoloOutputLayer> layer,
                       std::vector<HailoDetection> &objects);
};

void YoloPost::extract_boxes(std::shared_ptr<YoloOutputLayer> layer,
                             std::vector<HailoDetection> &objects)
{
    uint class_id = 0;
    float x, y, h, w, confidence, class_confidence = 0.0f;
    float xmin, ymin = 0.0f;
    for (uint row = 0; row < layer->_height; ++row)
    {
        for (uint col = 0; col < layer->_width; ++col)
        {
            for (uint anchor = 0; anchor < layer->NUM_ANCHORS; ++anchor)
            {
                confidence = layer->get_confidence(row, col, anchor);
                if (confidence < _detection_thr)
                    continue;
                std::tie(class_id, class_confidence) = layer->get_class(row, col, anchor);
                // Final confidence: box confidence * class probability
                confidence = confidence * class_confidence;
                if (confidence > _detection_thr)
                {
                    std::tie(x, y) = layer->get_center(row, col, anchor);
                    std::tie(w, h) = layer->get_shape(row, col, anchor, m_image_width, m_image_height);
                    // Get the top left corner of the object.
                    xmin = (x - (w / 2.0f));
                    ymin = (y - (h / 2.0f));
                    objects.push_back(HailoDetection(HailoBBox(xmin, ymin, w, h), class_id, m_dataset[(uint8_t)class_id], confidence));
                }
            }
        }
    }
}

class Yolov5 : public YoloPost
{
public:
    Yolov5(HailoROIPtr roi, YoloParams *params)
        : YoloPost(params->labels, params->detection_threshold, params->iou_threshold, params->max_boxes), _tensors(roi->get_tensors())
    {
        if (_tensors.size() > 0)
        {
            bool sigmoid = (params->output_activation == "sigmoid");
            sort(_tensors.begin(), _tensors.end(),
                 [](const HailoTensorPtr &a, const HailoTensorPtr &b)
                 { return a->size() < b->size(); });

            m_image_width = _tensors[0]->width() * 32;
            m_image_height = _tensors[0]->height() * 32;
            _layers.reserve(_tensors.size());
            for (std::size_t i = 0; i < _tensors.size(); i++)
            {
                hailo_format_type_t format = _tensors[i]->vstream_info().format.type;
                _layers.push_back(std::make_shared<Yolov5OL>(_tensors[i], params->anchors_vec[i], sigmoid, params->label_offset, format == HAILO_FORMAT_TYPE_UINT16));
            }
        }
        params->check_params_logic(get_num_classes());
    };

    virtual ~Yolov5() = default;

private:
    std::vector<HailoTensorPtr> _tensors;
};


void yolov5(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = Yolov5(roi, params);
    auto detections = post.decode();
    hailo_common::add_detections(roi, detections);
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    yolov5(roi, params);
}

YoloParams *init(const std::string config_path, const std::string function_name)
{
    YoloParams *params;
    if (!fs::exists(config_path))
    {
        std::cerr << "Config file doesn't exist, using default parameters" << std::endl;
        if (function_name.std::string::compare("yolov5") == 0)
        {
            params = new Yolov5Params;
        }
        if (function_name.std::string::compare("yolov7") == 0)
        {
            params = new Yolov7Params;
        }
        else
        {
            std::cerr << function_name << " network doesn't have default parameters, run might fail" << std::endl;
            params = new YoloParams;
        }
        return params;
    }
    else
    {
        params = new YoloParams;
        char config_buffer[4096];
        const char *json_schema = R""""({
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "iou_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
            },
            "detection_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
            },
            "output_activation": {
            "type": "string"
            },
            "label_offset": {
            "type": "integer"
            },
            "max_boxes": {
            "type": "integer"
            },
            "anchors": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                "type": "integer"
                }
            }
            },
            "labels": {
            "type": "array",
            "items": {
                "type": "string"
                }
            }
        },
        "required": [
            "iou_threshold",
            "detection_threshold",
            "output_activation",
            "label_offset",
            "max_boxes",
            "anchors",
            "labels"
        ]
        })"""";

        std::FILE *fp = fopen(config_path.c_str(), "r");
        if (fp == nullptr)
        {
            throw std::runtime_error("JSON config file is not valid");
        }
        rapidjson::FileReadStream stream(fp, config_buffer, sizeof(config_buffer));
        bool valid = common::validate_json_with_schema(stream, json_schema);
        if (valid)
        {
            rapidjson::Document doc_config_json;
            doc_config_json.ParseStream(stream);

            // parse labels
            auto labels = doc_config_json["labels"].GetArray();
            uint i = 0;
            for (auto &v : labels)
            {
                params->labels.insert(std::pair<std::uint8_t, std::string>(i, v.GetString()));
                i++;
            }
            // parse anchors
            auto config_anchors = doc_config_json["anchors"].GetArray();
            std::vector<std::vector<int>> anchors_vec;
            for (uint j = 0; j < config_anchors.Size(); j++)
            {
                uint size = config_anchors[j].GetArray().Size();
                std::vector<int> anchor;
                for (uint k = 0; k < size; k++)
                {
                    anchor.push_back(config_anchors[j].GetArray()[k].GetInt());
                }
                anchors_vec.push_back(anchor);
            }

            params->anchors_vec = anchors_vec;
            // set the params
            params->iou_threshold = doc_config_json["iou_threshold"].GetFloat();
            params->detection_threshold = doc_config_json["detection_threshold"].GetFloat();
            params->output_activation = doc_config_json["output_activation"].GetString();
            params->label_offset = doc_config_json["label_offset"].GetInt();
            params->max_boxes = doc_config_json["max_boxes"].GetInt();
            if (params->output_activation != "sigmoid" && params->output_activation != "none")
            {
                std::ostringstream oss;
                oss << "config output activation do not match! output activation: "
                    << params->output_activation << std::endl;
                throw std::runtime_error(oss.str());
            }
        }
        fclose(fp);
    }
    return params;
}
void YoloParams::check_params_logic(uint num_classes_tensors)
{
    if (labels.size() - 1 != num_classes_tensors)
    {
        std::ostringstream oss;
        oss << "config class labels do not match output tensors! config labels size: "
            << labels.size() - 1 << " tensors num classes: " << num_classes_tensors << std::endl;
        throw std::runtime_error(oss.str());
    }
}

void free_resources(void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    delete params;
}
