/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <typeinfo>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>

#include "common/yolo_postprocess.hpp"
#include "common/nms.hpp"
#include "common/labels/coco_eighty.hpp"
#include "common/json_config.hpp"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

// #if __GNUC__ > 8
// #include <filesystem>
// namespace fs = std::filesystem;
// #else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
// #endif

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
                    if (objects.size() < _max_boxes)
                        objects.push_back(HailoDetection(HailoBBox(xmin, ymin, w, h), class_id, m_dataset[class_id], confidence));
                    else
                        return;
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
        bool sigmoid = (params->output_activation == "sigmoid");
        yolov5_init(params->anchors_vec, sigmoid, params->label_offset);
        params->check_params_logic(get_num_classes());      
    };

    void yolov5_init(std::vector<std::vector<int>> anchors, bool perform_sigmoid, int label_offset)
    {
        if (_tensors.size() > 0)
        {
            sort(_tensors.begin(), _tensors.end(),
                 [](const HailoTensorPtr &a, const HailoTensorPtr &b)
                 { return a->size() < b->size(); });

            m_image_width = _tensors[0]->width() * 32;
            m_image_height = _tensors[0]->height() * 32;
            _layers.reserve(_tensors.size());
            for (std::size_t i = 0; i < _tensors.size(); i++)
            {
                _layers.push_back(std::make_shared<Yolov5OL>(_tensors[i], anchors[i], perform_sigmoid, label_offset));
            }
        }
    }

    virtual ~Yolov5() = default;

private:
    std::vector<HailoTensorPtr> _tensors;
};

class YoloSplitted : public YoloPost
{
public:
    virtual ~YoloSplitted() = default;

    YoloSplitted(HailoROIPtr roi,
                 std::map<uint8_t, std::string> dataset,
                 std::vector<std::vector<int>> anchors,
                 float detection_threshold,
                 float iou_threshold,
                 uint max_boxes)
        : YoloPost(dataset, detection_threshold, iou_threshold, max_boxes), _roi(roi){};

protected:
    HailoROIPtr _roi;
};

class Yolov3 : public YoloSplitted
{
public:
    Yolov3(HailoROIPtr roi, YoloParams *params)
        : YoloSplitted(roi, params->labels, params->anchors_vec, params->detection_threshold, params->iou_threshold, params->max_boxes)
    {
        if (roi->has_tensors())
        {
            auto anchors = params->anchors_vec;
            m_image_width = _roi->get_tensor("yolov3/conv83_centers")->width() * 32;
            m_image_height = _roi->get_tensor("yolov3/conv83_centers")->height() * 32;
            _layers.push_back(std::make_shared<Yolov3OL>(_roi->get_tensor("yolov3/conv83_centers"), _roi->get_tensor("yolov3/conv83_scales"),
                                                         _roi->get_tensor("yolov3/conv83_obj"), _roi->get_tensor("yolov3/conv83_probs"),
                                                         anchors[0], params->label_offset));
            _layers.push_back(std::make_shared<Yolov3OL>(_roi->get_tensor("yolov3/conv91_centers"), _roi->get_tensor("yolov3/conv91_scales"),
                                                         _roi->get_tensor("yolov3/conv91_obj"), _roi->get_tensor("yolov3/conv91_probs"),
                                                         anchors[1], params->label_offset));
            _layers.push_back(std::make_shared<Yolov3OL>(_roi->get_tensor("yolov3/conv98_centers"), _roi->get_tensor("yolov3/conv98_scales"),
                                                         _roi->get_tensor("yolov3/conv98_obj"), _roi->get_tensor("yolov3/conv98_probs"),
                                                         anchors[2], params->label_offset));
        }
    };

    virtual ~Yolov3() = default;
};

class Yolov4 : public YoloSplitted
{
public:
    Yolov4(HailoROIPtr roi, YoloParams *params)
        : YoloSplitted(roi, params->labels, params->anchors_vec, params->detection_threshold, params->iou_threshold, params->max_boxes)
    {
        if (_roi->has_tensors())
        {
            auto anchors = params->anchors_vec;
            m_image_width = _roi->get_tensor("yolov4_leaky/conv110_centers")->width() * 32;
            m_image_height = _roi->get_tensor("yolov4_leaky/conv110_centers")->height() * 32;
            _layers.push_back(std::make_shared<Yolov4OL>(_roi->get_tensor("yolov4_leaky/conv110_centers"), _roi->get_tensor("yolov4_leaky/conv110_scales"),
                                                         _roi->get_tensor("yolov4_leaky/conv110_obj"), _roi->get_tensor("yolov4_leaky/conv110_probs"),
                                                         anchors[0], params->label_offset));
            _layers.push_back(std::make_shared<Yolov4OL>(_roi->get_tensor("yolov4_leaky/conv103_centers"), _roi->get_tensor("yolov4_leaky/conv103_scales"),
                                                         _roi->get_tensor("yolov4_leaky/conv103_obj"), _roi->get_tensor("yolov4_leaky/conv103_probs"),
                                                         anchors[1], params->label_offset));
            _layers.push_back(std::make_shared<Yolov4OL>(_roi->get_tensor("yolov4_leaky/conv95_centers"), _roi->get_tensor("yolov4_leaky/conv95_scales"),
                                                         _roi->get_tensor("yolov4_leaky/conv95_obj"), _roi->get_tensor("yolov4_leaky/conv95_probs"),
                                                         anchors[2], params->label_offset));

            params->check_params_logic(get_num_classes());
        }
    };
    virtual ~Yolov4() = default;
};

class TinyYolov4LicensePlates : public YoloSplitted
{
public:
    TinyYolov4LicensePlates(HailoROIPtr roi, YoloParams *params)
        : YoloSplitted(roi, params->labels, params->anchors_vec, params->detection_threshold, params->iou_threshold, params->max_boxes)
    {
        if (_roi->has_tensors())
        {
            auto anchors = params->anchors_vec;
            m_image_width = _roi->get_tensor("tiny_yolov4_license_plates/conv19_centers")->width() * 32;
            m_image_height = _roi->get_tensor("tiny_yolov4_license_plates/conv19_centers")->height() * 32;
            _layers.push_back(std::make_shared<Yolov4OL>(_roi->get_tensor("tiny_yolov4_license_plates/conv19_centers"), _roi->get_tensor("tiny_yolov4_license_plates/conv19_scales"),
                                                         _roi->get_tensor("tiny_yolov4_license_plates/conv19_obj"), _roi->get_tensor("tiny_yolov4_license_plates/conv19_probs"),
                                                         anchors[0], params->label_offset));
            _layers.push_back(std::make_shared<Yolov4OL>(_roi->get_tensor("tiny_yolov4_license_plates/conv21_centers"), _roi->get_tensor("tiny_yolov4_license_plates/conv21_scales"),
                                                         _roi->get_tensor("tiny_yolov4_license_plates/conv21_obj"), _roi->get_tensor("tiny_yolov4_license_plates/conv21_probs"),
                                                         anchors[1], params->label_offset));
            params->check_params_logic(get_num_classes());
        }
    };

    virtual ~TinyYolov4LicensePlates() = default;
};

class YoloX : public YoloPost
{
public:
    YoloX(HailoROIPtr roi, YoloParams *params)
        : YoloPost(params->labels, params->detection_threshold, params->iou_threshold, params->max_boxes), _roi(roi)
    {
        if (_roi->has_tensors())
        {
            m_image_width = _roi->get_tensor("yolox_l_leaky/conv130")->width() * 32;
            m_image_height = _roi->get_tensor("yolox_l_leaky/conv130")->height() * 32;

            _layers.push_back(std::make_shared<YoloXOL>(_roi->get_tensor("yolox_l_leaky/conv130"), _roi->get_tensor("yolox_l_leaky/conv131"),
                                                        _roi->get_tensor("yolox_l_leaky/conv129"), params->label_offset));
            _layers.push_back(std::make_shared<YoloXOL>(_roi->get_tensor("yolox_l_leaky/conv113"), _roi->get_tensor("yolox_l_leaky/conv114"),
                                                        _roi->get_tensor("yolox_l_leaky/conv112"), params->label_offset));
            _layers.push_back(std::make_shared<YoloXOL>(_roi->get_tensor("yolox_l_leaky/conv95"), _roi->get_tensor("yolox_l_leaky/conv96"),
                                                        _roi->get_tensor("yolox_l_leaky/conv94"), params->label_offset));
            params->check_params_logic(get_num_classes());
        }
    };
    ~YoloX() = default;

protected:
    HailoROIPtr _roi;
};

void yolov5_no_persons(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = Yolov5(roi, params);
    auto detections = post.decode();
    int person_class_id = 1;
    detections.erase(std::remove_if(detections.begin(), detections.end(),
                                    [person_class_id](HailoDetection obj)
                                    { return obj.get_class_id() == person_class_id; }),
                     detections.end());
    hailo_common::add_detections(roi, detections);
}

void yolov5_personface(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());

    // Yolov5 Postprocess for faces
    auto post = Yolov5(roi, params);
    auto face_detections = post.decode();
    auto face_class_id = 2;
    // Take only the faces.
    face_detections.erase(std::remove_if(face_detections.begin(), face_detections.end(),
                                         [face_class_id](HailoDetection obj)
                                         { return obj.get_class_id() != face_class_id; }),
                          face_detections.end());


    // Fix the bboxes because resize was letterbox.
    for (auto &det : face_detections)
    {
        auto detection_bbox = det.get_bbox();
        auto xmin = (detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymin = (detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin();
        auto xmax = (detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymax = (detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin();

        HailoBBox new_bbox(xmin, ymin, xmax - xmin, ymax - ymin);
        det.set_bbox(new_bbox);
 //       std::cout << "face bbox xmin is: " << xmin << " ymin is: " << ymin << " xmax is: " << xmax << " ymax is: " << ymax << std::endl;
    }

    // Yolov5 Postprocess for persons
    float person_detection_thr = 0.5f;
    auto person_class_id = 1;
    params->detection_threshold = person_detection_thr;
    auto persons_post = Yolov5(roi, params);
    auto person_detections = persons_post.decode();
 //   std::cout << "person detections size is: " << person_detections.size() << std::endl;
    
    // Take only the persons.
    person_detections.erase(std::remove_if(person_detections.begin(), person_detections.end(),
                                           [person_class_id](HailoDetection obj)
                                           { return obj.get_class_id() != person_class_id; }),
                            person_detections.end());

    // Fix the bboxes because resize was letterbox.
    for (auto &det : person_detections)
    {
        auto detection_bbox = det.get_bbox();
        auto xmin = (detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymin = (detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin();
        auto xmax = (detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin();
        auto ymax = (detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin();

        HailoBBox new_bbox(xmin, ymin, xmax - xmin, ymax - ymin);
        det.set_bbox(new_bbox);
        //std::cout << "person bbox xmin is: " << xmin << " ymin is: " << ymin << " xmax is: " << xmax << " ymax is: " << ymax << std::endl;
    }

    // Clear the scaling bbox of main roi because all detections are fixed.
    roi->clear_scaling_bbox();

    // Add detections to main roi.
    hailo_common::add_detections(roi, face_detections);
    hailo_common::add_detections(roi, person_detections);
}

void yolov5_vehicles_only(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = Yolov5(roi, params);
    auto detections = post.decode();
    hailo_common::add_detections(roi, detections);
}

void yolov5(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = Yolov5(roi, params);
    auto detections = post.decode();
    hailo_common::add_detections(roi, detections);
}

void yolov3(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = Yolov3(roi, params);
    auto detections = post.decode();
    hailo_common::add_detections(roi, detections);
}

void yolov4(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = Yolov4(roi, params);
    auto detections = post.decode();
    hailo_common::add_detections(roi, detections);
}

void tiny_yolov4_license_plates(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = TinyYolov4LicensePlates(roi, params);
    auto detections = post.decode();
    hailo_common::add_detections(roi, detections);
}

void yolox(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    auto post = YoloX(roi, params);
    auto detections = post.decode();
    hailo_common::add_detections(roi, detections);
}

void filter(HailoROIPtr roi, void *params_void_ptr)
{
    YoloParams *params = reinterpret_cast<YoloParams *>(params_void_ptr);
    yolov5(roi, params);
}

YoloParams *init(const std::string config_path)
{
    YoloParams *params = new YoloParams;
    if (!fs::exists(config_path))
    {
        std::cerr << "Config file doesn't exist, using default parameters" << std::endl;
        return params;
    }
    else
    {
        // TODO edit schema to check all params in json
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

// int main() {return 0;}
