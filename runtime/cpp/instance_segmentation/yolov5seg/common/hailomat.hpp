/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file overlay/common.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-01-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <opencv2/opencv.hpp>
#include "hailo_common.hpp"
#include "hailo_objects.hpp"

// Transformations were taken from https://stackoverflow.com/questions/17892346/how-to-convert-rgb-yuv-rgb-both-ways.
#define RGB2Y(R, G, B) CLIP((0.257 * (R) + 0.504 * (G) + 0.098 * (B)) + 16)
#define RGB2U(R, G, B) CLIP((-0.148 * (R)-0.291 * (G) + 0.439 * (B)) + 128)
#define RGB2V(R, G, B) CLIP((0.439 * (R)-0.368 * (G)-0.071 * (B)) + 128)

typedef enum
{
    HAILO_MAT_NONE = -1,
    HAILO_MAT_RGB,
    HAILO_MAT_RGBA,
    HAILO_MAT_YUY2,
    HAILO_MAT_NV12
} hailo_mat_t;

typedef enum
{
    NONE = -1,
    VERTICAL,
    HORIZONTAL,
    DIAGONAL,
    ANTI_DIAGONAL,
} LineOrientation;

inline LineOrientation line_orientation(cv::Point point1, cv::Point point2)
{
    if (point1.x == point2.x)
        return VERTICAL;
    else if (point1.y == point2.y)
        return HORIZONTAL;
    else if (point1.x < point2.x && point1.y < point2.y)
        return DIAGONAL;
    else if (point1.x < point2.x && point1.y > point2.y)
        return ANTI_DIAGONAL;
    else
        return NONE;
}

inline int floor_to_even_number(int x)
{
    /*
    The expression x &~1 in C++ performs a bitwise AND operation between the number x and the number ~1(bitwise negation of 1).
    In binary representation, the number 1 is represented as 0000 0001, and its negation, ~1, is equal to 1111 1110.
    The bitwise AND operation between x and ~1 zeros out the least significant bit of x,
    effectively rounding it down to the nearest even number.
    This is because any odd number in binary representation will have its least significant bit set to 1,
    and ANDing it with ~1 will zero out that bit.
    */
    return x & ~1;
}

class HailoMat
{
protected:
    uint m_height;
    uint m_width;
    uint m_native_height;
    uint m_native_width;
    uint m_stride;
    int m_line_thickness;
    int m_font_thickness;
    cv::Rect get_bounding_rect(HailoBBox bbox, uint channel_width, uint channel_height)
    {
        cv::Rect rect;
        uint width = channel_width;
        uint height = channel_height;
        rect.x = CLAMP(bbox.xmin() * width, 0, width);
        rect.y = CLAMP(bbox.ymin() * height, 0, height);
        rect.width = CLAMP(bbox.width() * width, 0, width - rect.x);
        rect.height = CLAMP(bbox.height() * height, 0, height - rect.y);
        return rect;
    }

public:
    HailoMat(uint height, uint width, uint stride, int line_thickness = 1, int font_thickness = 1) : m_height(height),
                                                                                                     m_width(width),
                                                                                                     m_native_height(height),
                                                                                                     m_native_width(width),
                                                                                                     m_stride(stride),
                                                                                                     m_line_thickness(line_thickness),
                                                                                                     m_font_thickness(font_thickness){};
    HailoMat() : m_height(0), m_width(0), m_native_height(0), m_native_width(0), m_stride(0), m_line_thickness(0), m_font_thickness(0){};
    virtual ~HailoMat() = default;
    uint width() { return m_width; };
    uint height() { return m_height; };
    uint native_width() { return m_native_width; };
    uint native_height() { return m_native_height; };
    virtual cv::Mat &get_mat() = 0;
    virtual void draw_rectangle(cv::Rect rect, const cv::Scalar color) = 0;
    virtual void draw_text(std::string text, cv::Point position, double font_scale, const cv::Scalar color) = 0;
    virtual void draw_line(cv::Point point1, cv::Point point2, const cv::Scalar color, int thickness, int line_type) = 0;
    virtual void draw_ellipse(cv::Point center, cv::Size axes, double angle, double start_angle, double end_angle, const cv::Scalar color, int thickness) = 0;
    virtual void blur(cv::Rect rect, cv::Size ksize) = 0;
    /*
     * @brief Crop ROIs from the mat, note the present implementation is valid
     *        for interlaced formats. Planar formats such as NV12 should override.
     *
     * @param crop_roi
     *        The roi to crop from this mat.
     * @return cv::Mat
     *         The cropped mat.
     */
    virtual cv::Mat crop(HailoROIPtr crop_roi)
    {
        cv::Rect rect = get_crop_rect(crop_roi);
        cv::Mat cropped_cv_mat = get_mat()(rect);
        return cropped_cv_mat;
    }

    virtual cv::Rect get_crop_rect(HailoROIPtr crop_roi)
    {
        auto bbox = hailo_common::create_flattened_bbox(crop_roi->get_bbox(), crop_roi->get_scaling_bbox());
        cv::Rect rect = get_bounding_rect(bbox, m_width, m_height);
        return rect;
    }

    /**
     * @brief Get the type of mat
     *
     * @return hailo_mat_t - The type of the mat.
     */
    virtual hailo_mat_t get_type() = 0;
};

class HailoRGBMat : public HailoMat
{
protected:
    cv::Mat m_mat;
    std::string m_name;

public:
    HailoRGBMat(uint8_t *buffer, uint height, uint width, uint stride, int line_thickness = 1, int font_thickness = 1, std::string name = "HailoRGBMat") : HailoMat(height, width, stride, line_thickness, font_thickness)
    {
        m_name = name;
        m_mat = cv::Mat(m_height, m_width, CV_8UC3, buffer, m_stride);
    };
    HailoRGBMat(cv::Mat mat, std::string name, int line_thickness = 1, int font_thickness = 1)
    {
        m_mat = mat;
        m_name = name;
        m_height = mat.rows;
        m_width = mat.cols;
        m_stride = mat.step;
        m_native_height = m_height;
        m_native_width = m_width;
        m_line_thickness = line_thickness;
        m_font_thickness = font_thickness;
    }
    virtual hailo_mat_t get_type()
    {
        return HAILO_MAT_RGB;
    }
    virtual cv::Mat &get_mat()
    {
        return m_mat;
    }
    virtual std::string get_name() const
    {
        return m_name;
    }
    virtual void draw_rectangle(cv::Rect rect, const cv::Scalar color)
    {
        cv::rectangle(m_mat, rect, color, m_line_thickness);
    }
    virtual void draw_text(std::string text, cv::Point position, double font_scale, const cv::Scalar color)
    {
        cv::putText(m_mat, text, position, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, m_font_thickness);
    }
    virtual void draw_line(cv::Point point1, cv::Point point2, const cv::Scalar color, int thickness, int line_type)
    {
        cv::line(m_mat, point1, point2, color, thickness, line_type);
    }
    virtual void draw_ellipse(cv::Point center, cv::Size axes, double angle, double start_angle, double end_angle, const cv::Scalar color, int thickness)
    {
        cv::ellipse(m_mat, center, axes, angle, start_angle, end_angle, color, thickness);
    }
    virtual void blur(cv::Rect rect, cv::Size ksize)
    {
        cv::Mat target_roi = this->m_mat(rect);
        cv::blur(target_roi, target_roi, ksize);
    }
    virtual ~HailoRGBMat()
    {
        m_mat.release();
    }
};

class HailoRGBAMat : public HailoMat
{
protected:
    cv::Mat m_mat;
    cv::Scalar get_rgba_color(cv::Scalar rgb_color, int alpha = 1.0)
    {
        // setting default alpha as 1.0 as shown in an example: https://www.w3schools.com/css/css_colors_rgb.asp
        rgb_color[3] = alpha;
        return rgb_color;
    }

public:
    HailoRGBAMat(uint8_t *buffer, uint height, uint width, uint stride, int line_thickness = 1, int font_thickness = 1) : HailoMat(height, width, stride, line_thickness, font_thickness)
    {
        m_mat = cv::Mat(m_height, m_width, CV_8UC4, buffer, m_stride);
    };
    virtual hailo_mat_t get_type()
    {
        return HAILO_MAT_RGBA;
    }
    virtual cv::Mat &get_mat()
    {
        return m_mat;
    }
    virtual void draw_rectangle(cv::Rect rect, const cv::Scalar color)
    {
        cv::rectangle(m_mat, rect, get_rgba_color(color), m_line_thickness);
    }
    virtual void draw_text(std::string text, cv::Point position, double font_scale, const cv::Scalar color)
    {
        cv::putText(m_mat, text, position, cv::FONT_HERSHEY_SIMPLEX, font_scale, get_rgba_color(color), m_font_thickness);
    }
    virtual void draw_line(cv::Point point1, cv::Point point2, const cv::Scalar color, int thickness, int line_type)
    {
        cv::line(m_mat, point1, point2, get_rgba_color(color), thickness, line_type);
    }
    virtual void draw_ellipse(cv::Point center, cv::Size axes, double angle, double start_angle, double end_angle, const cv::Scalar color, int thickness)
    {
        cv::ellipse(m_mat, center, axes, angle, start_angle, end_angle, get_rgba_color(color), thickness);
    }
    virtual void blur(cv::Rect rect, cv::Size ksize)
    {
        cv::Mat target_roi = this->m_mat(rect);
        cv::blur(target_roi, target_roi, ksize);
    }
    virtual ~HailoRGBAMat()
    {
        m_mat.release();
    }
};

class HailoYUY2Mat : public HailoMat
{
protected:
    cv::Mat m_mat;
    cv::Scalar get_yuy2_color(cv::Scalar rgb_color)
    {
        uint r = rgb_color[0];
        uint g = rgb_color[1];
        uint b = rgb_color[2];
        uint y = RGB2Y(r, g, b);
        uint u = RGB2U(r, g, b);
        uint v = RGB2V(r, g, b);
        return cv::Scalar(y, u, y, v);
    }

public:
    HailoYUY2Mat(uint8_t *buffer, uint height, uint width, uint stride, int line_thickness = 1, int font_thickness = 1) : HailoMat(height, width, stride, line_thickness, font_thickness)
    {
        m_width = m_width / 2;
        m_mat = cv::Mat(m_height, m_width, CV_8UC4, buffer, m_stride);
    };
    virtual hailo_mat_t get_type()
    {
        return HAILO_MAT_YUY2;
    }
    virtual cv::Mat &get_mat()
    {
        return m_mat;
    }
    virtual void draw_rectangle(cv::Rect rect, const cv::Scalar color)
    {
        cv::Rect fixed_rect = cv::Rect(rect.x / 2, rect.y, rect.width / 2, rect.height);
        cv::rectangle(m_mat, fixed_rect, get_yuy2_color(color), m_line_thickness);
    }
    virtual void draw_text(std::string text, cv::Point position, double font_scale, const cv::Scalar color){};
    virtual void draw_line(cv::Point point1, cv::Point point2, const cv::Scalar color, int thickness, int line_type){};
    virtual void draw_ellipse(cv::Point center, cv::Size axes, double angle, double start_angle, double end_angle, const cv::Scalar color, int thickness){};
    virtual void blur(cv::Rect rect, cv::Size ksize){};
    virtual ~HailoYUY2Mat()
    {
        m_mat.release();
    }
};

class HailoNV12Mat : public HailoMat
{
    /**
        NV12 Layout in memory (planar YUV 4:2:0):

        +-----+-----+-----+-----+-----+-----+
        | Y0  | Y1  | Y2  | Y3  | Y4  | Y5  |
        +-----+-----+-----+-----+-----+-----+
        | Y6  | ... | ... | ... | ... | ... |
        +-----+-----+-----+-----+-----+-----+
        | Y12 | ... | ... | ... | ... | ... |
        +-----+-----+-----+-----+-----+-----+
        | Y18 | ... | ... | ... | ... | ... |
        +-----+-----+-----+-----+-----+-----+
        | Y24 | ... | ... | ... | ... | ... |
        +-----+-----+-----+-----+-----+-----+
        | Y30 | Y31 | Y32 | Y33 | Y34 | Y35 |
        +-----+-----+-----+-----+-----+-----+
        | U0  | V0  | U1  | V1  | U2  | V2  |
        +-----+-----+-----+-----+-----+-----+
        | U3  | V3  | ... | ... | ... | ... |
        +-----+-----+-----+-----+-----+-----+
        | U6  | V6  | U7  | V7  | U8  | V8  |
        +-----+-----+-----+-----+-----+-----+
    */
protected:
    cv::Mat m_mat;
    cv::Mat m_y_plane_mat;
    cv::Mat m_uv_plane_mat;

    cv::Scalar get_nv12_color(cv::Scalar rgb_color)
    {
        uint r = rgb_color[0];
        uint g = rgb_color[1];
        uint b = rgb_color[2];
        uint y = RGB2Y(r, g, b);
        uint u = RGB2U(r, g, b);
        uint v = RGB2V(r, g, b);
        return cv::Scalar(y, u, v);
    }

public:
    HailoNV12Mat(uint8_t *buffer, uint height, uint width, uint y_plane_stride, uint uv_plane_stride, int line_thickness = 1, int font_thickness = 1, void *plane0 = nullptr, void *plane1 = nullptr) : HailoMat(height, width, y_plane_stride, line_thickness, font_thickness)
    {
        m_height = (m_height * 3 / 2);
        m_mat = cv::Mat(m_height, m_width, CV_8UC1, buffer, y_plane_stride);

        if (plane0 == nullptr)
            plane0 = (char *)m_mat.data;
        if (plane1 == nullptr)
            plane1 = (char *)m_mat.data + ((m_native_height)*m_native_width);


        m_y_plane_mat = cv::Mat(m_native_height, m_width, CV_8UC1, plane0, y_plane_stride);
        m_uv_plane_mat = cv::Mat(m_native_height / 2, m_native_width / 2, CV_8UC2, plane1, uv_plane_stride);
    };
    virtual hailo_mat_t get_type()
    {
        return HAILO_MAT_NV12;
    }
    virtual cv::Mat &get_mat()
    {
        return m_mat;
    }
    virtual void draw_rectangle(cv::Rect rect, const cv::Scalar color)
    {
        cv::Scalar yuv_color = get_nv12_color(color);
        uint thickness = m_line_thickness > 1 ? m_line_thickness / 2 : 1;
        // always floor the rect coordinates to even numbers to avoid drawing on the wrong pixel
        int y_plane_rect_x = floor_to_even_number(rect.x);
        int y_plane_rect_y = floor_to_even_number(rect.y);
        int y_plane_rect_width = floor_to_even_number(rect.width);
        int y_plane_rect_height = floor_to_even_number(rect.height);

        cv::Rect y1_rect = cv::Rect(y_plane_rect_x, y_plane_rect_y, y_plane_rect_width, y_plane_rect_height);
        cv::Rect y2_rect = cv::Rect(y_plane_rect_x + 1, y_plane_rect_y + 1, y_plane_rect_width - 2, y_plane_rect_height - 2);
        cv::rectangle(m_y_plane_mat, y1_rect, cv::Scalar(yuv_color[0]), thickness);
        cv::rectangle(m_y_plane_mat, y2_rect, cv::Scalar(yuv_color[0]), thickness);

        cv::Rect uv_rect = cv::Rect(y_plane_rect_x / 2, y_plane_rect_y / 2, y_plane_rect_width / 2, y_plane_rect_height / 2);
        cv::rectangle(m_uv_plane_mat, uv_rect, cv::Scalar(yuv_color[1], yuv_color[2]), thickness);
    }

    virtual void draw_text(std::string text, cv::Point position, double font_scale, const cv::Scalar color)
    {
        cv::Scalar yuv_color = get_nv12_color(color);
        cv::Point y_position = cv::Point(position.x, position.y);
        cv::Point uv_position = cv::Point(position.x / 2, position.y / 2);
        cv::putText(m_y_plane_mat, text, y_position, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(yuv_color[0]), m_font_thickness);
        cv::putText(m_uv_plane_mat, text, uv_position, cv::FONT_HERSHEY_SIMPLEX, font_scale / 2, cv::Scalar(yuv_color[1], yuv_color[2]), m_font_thickness / 2);
    };

    virtual void draw_line(cv::Point point1, cv::Point point2, const cv::Scalar color, int thickness, int line_type)
    {
        cv::Scalar yuv_color = get_nv12_color(color);

        int y_plane_x1_value = floor_to_even_number(point1.x);
        int y_plane_y1_value = floor_to_even_number(point1.y);
        int y_plane_x2_value = floor_to_even_number(point2.x);
        int y_plane_y2_value = floor_to_even_number(point2.y);

        cv::line(m_uv_plane_mat, cv::Point(y_plane_x1_value / 2, y_plane_y1_value / 2), cv::Point(y_plane_x2_value / 2, y_plane_y2_value / 2), cv::Scalar(yuv_color[1], yuv_color[2]), thickness, line_type);

        switch (line_orientation(point1, point2))
        {
        case HORIZONTAL:
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value, y_plane_y1_value), cv::Point(y_plane_x2_value, y_plane_y2_value), cv::Scalar(yuv_color[0]), thickness, line_type);
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value, y_plane_y1_value + 1), cv::Point(y_plane_x2_value, y_plane_y2_value + 1), cv::Scalar(yuv_color[0]), thickness, line_type);
            break;
        case VERTICAL:
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value, y_plane_y1_value), cv::Point(y_plane_x2_value, y_plane_y2_value), cv::Scalar(yuv_color[0]), thickness, line_type);
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value + 1, y_plane_y1_value), cv::Point(y_plane_x2_value + 1, y_plane_y2_value), cv::Scalar(yuv_color[0]), thickness, line_type);
            break;
        case DIAGONAL:
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value, y_plane_y1_value), cv::Point(y_plane_x2_value, y_plane_y2_value), cv::Scalar(yuv_color[0]), thickness, line_type);
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value + 1, y_plane_y1_value + 1), cv::Point(y_plane_x2_value + 1, y_plane_y2_value + 1), cv::Scalar(yuv_color[0]), thickness, line_type);
            break;
        case ANTI_DIAGONAL:
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value, y_plane_y1_value), cv::Point(y_plane_x2_value, y_plane_y2_value), cv::Scalar(yuv_color[0]), thickness, line_type);
            cv::line(m_y_plane_mat, cv::Point(y_plane_x1_value + 1, y_plane_y1_value), cv::Point(y_plane_x2_value + 1, y_plane_y2_value), cv::Scalar(yuv_color[0]), thickness, line_type);
            break;
        default:
            break;
        }
    }

    virtual void draw_ellipse(cv::Point center, cv::Size axes, double angle, double start_angle, double end_angle, const cv::Scalar color, int thickness)
    {
        thickness = 2;
        // Wrap the mat with Y and UV channel windows
        cv::Scalar yuv_color = get_nv12_color(color);

        cv::Point y_position = cv::Point(floor_to_even_number(center.x), floor_to_even_number(center.y));
        cv::Point uv_position = cv::Point(floor_to_even_number(center.x) / 2, floor_to_even_number(center.y) / 2);

        cv::ellipse(m_y_plane_mat, y_position, {floor_to_even_number(axes.width), floor_to_even_number(axes.height)}, angle, start_angle, end_angle, cv::Scalar(yuv_color[0]), thickness / 2);
        cv::ellipse(m_y_plane_mat, y_position, {floor_to_even_number(axes.width) + 1, floor_to_even_number(axes.height) + 1}, angle, start_angle, end_angle, cv::Scalar(yuv_color[0]), thickness / 2);
        cv::ellipse(m_uv_plane_mat, uv_position, axes / 2, angle, start_angle, end_angle, cv::Scalar(yuv_color[1], yuv_color[2]), thickness);
    }

    virtual void blur(cv::Rect rect, cv::Size ksize)
    {
        cv::Rect y_rect = cv::Rect(rect.x, rect.y, rect.width, rect.height);
        cv::Mat target_roi = this->m_mat(y_rect);
        cv::blur(target_roi, target_roi, ksize);
    }

    virtual cv::Rect get_crop_rect(HailoROIPtr crop_roi)
    {
        auto bbox = hailo_common::create_flattened_bbox(crop_roi->get_bbox(), crop_roi->get_scaling_bbox());

        // The Y channel is packed before the U & V on it's own
        cv::Rect y_rect = get_bounding_rect(bbox, m_native_width, m_native_height);
        // y_rect values should be even (round if necessary)
        y_rect.width = floor_to_even_number(y_rect.width);
        y_rect.height = floor_to_even_number(y_rect.height);
        y_rect.x = floor_to_even_number(y_rect.x);
        y_rect.y = floor_to_even_number(y_rect.y);

        return y_rect;
    }

    virtual cv::Mat crop(HailoROIPtr crop_roi)
    {
        // Wrap the mat with Y and UV channel windows
        cv::Rect y_rect = get_crop_rect(crop_roi);

        // The U and V channels are interlaced together after the Y channel,
        // so they need to be cropped separately
        cv::Rect uv_rect;
        // uv_rect values should be exactly half the size of y_rect
        uv_rect.width = y_rect.width / 2;
        uv_rect.height = y_rect.height / 2;
        uv_rect.x = y_rect.x / 2;
        uv_rect.y = y_rect.y / 2;

        // Prepare the cropped mat, and channel windows to fill with Y, U , & V
        cv::Mat cropped_mat = cv::Mat(y_rect.height + uv_rect.height, y_rect.width, CV_8UC1, m_stride);
        cv::Mat cropped_y_mat = cv::Mat(y_rect.height, y_rect.width, CV_8UC1, (char *)cropped_mat.data, y_rect.width);
        cv::Mat cropped_uv_mat = cv::Mat(uv_rect.height, uv_rect.width, CV_8UC2, (char *)cropped_mat.data + (y_rect.height * y_rect.width), y_rect.width);

        // Fill the cropped mat with the cropped channels
        m_y_plane_mat(y_rect).copyTo(cropped_y_mat);
        m_uv_plane_mat(uv_rect).copyTo(cropped_uv_mat);
        return cropped_mat;
    }
    virtual ~HailoNV12Mat()
    {
        m_y_plane_mat.release();
        m_uv_plane_mat.release();
        m_mat.release();
    }
};