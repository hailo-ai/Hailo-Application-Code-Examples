/**
 * Copyright 2020 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/
/**
 * @ file example_common.h
 * Common macros and defines used by Hailort Examples
 **/

#ifndef _EXAMPLE_COMMON_H_
#define _EXAMPLE_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include "hailo/hailort.h"
#include "double_buffer.hpp"
#include <map>


#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */


class FeatureData {
public:
    FeatureData(uint32_t buffers_size, float32_t qp_zp, float32_t qp_scale, uint32_t height, uint32_t width, uint32_t channels) :
    m_buffers(buffers_size), m_qp_zp(qp_zp), m_qp_scale(qp_scale), m_height(height), m_width(width), m_channels(channels)
    {}
    static bool sort_tensors_by_size (std::shared_ptr<FeatureData> i, std::shared_ptr<FeatureData> j) { 
		if(i->m_width == j->m_width){
			return i->m_channels < j->m_channels;
		}
		return i->m_width > j->m_width; 
	};

    DoubleBuffer m_buffers;
    float32_t m_qp_zp;
    float32_t m_qp_scale;
	uint32_t m_height;
	uint32_t m_width;
	uint32_t m_channels;
};


static inline std::map<uint8_t, std::string> coco_ninety_classes = {
        {0, "unlabeled"},
        {1, "person"},
        {10, "traffic light"},
        {11, "fire hydrant"},
        {13, "stop sign"},
        {14, "parking meter"},
        {15, "bench"},
        {16, "bird"},
        {17, "cat"},
        {18, "dog"},
        {19, "horse"},
        {20, "sheep"},
        {2, "bicycle"},
        {21, "cow"},
        {22, "elephant"},
        {23, "bear"},
        {24, "zebra"},
        {25, "giraffe"},
        {27, "backpack"},
        {28, "umbrella"},
        {31, "handbag"},
        {32, "tie"},
        {33, "suitcase"},
        {3, "car"},
        {34, "frisbee"},
        {35, "skis"},
        {36, "snowboard"},
        {37, "sports ball"},
        {38, "kite"},
        {39, "baseball bat"},
        {40, "baseball glove"},
        {41, "skateboard"},
        {42, "surfboard"},
        {43, "tennis racket"},
        {4, "motorcycle"},
        {44, "bottle"},
        {46, "wine glass"},
        {47, "cup"},
        {48, "fork"},
        {49, "knife"},
        {50, "spoon"},
        {51, "bowl"},
        {52, "banana"},
        {53, "apple"},
        {54, "sandwich"},
        {5, "airplane"},
        {55, "orange"},
        {56, "broccoli"},
        {57, "carrot"},
        {58, "hot dog"},
        {59, "pizza"},
        {60, "donut"},
        {61, "cake"},
        {62, "chair"},
        {63, "couch"},
        {64, "potted plant"},
        {6, "bus"},
        {65, "bed"},
        {67, "dining table"},
        {70, "toilet"},
        {72, "tv"},
        {73, "laptop"},
        {74, "mouse"},
        {75, "remote"},
        {76, "keyboard"},
        {77, "cellphone"},
        {78, "microwave"},
        {7, "train"},
        {79, "oven"},
        {80, "toaster"},
        {81, "sink"},
        {82, "refrigerator"},
        {84, "book"},
        {85, "clock"},
        {86, "vase"},
        {87, "scissors"},
        {88, "teddy bear"},
        {89, "hair drier"},
        {8, "truck"},
        {90, "tooth brush"},
        {9, "boat"}};


inline std::string get_coco17_name_from_int(int cls){
	return coco_ninety_classes[uint8_t(cls)];
}


inline std::string get_coco14_name_from_int(int cls)
{
    std::string result = "N/A";
    switch(cls) {
		case 0: result = "__background__";break;
		case 1: result = "person";break;
		case 2: result = "bicycle";break;
		case 3: result = "car";break;
		case 4: result = "motorcycle";break;
		case 5: result = "airplane";break;
		case 6: result = "bus";break;
		case 7: result = "train";break;
		case 8: result = "truck";break;
		case 9: result = "boat";break;
		case 10: result = "traffic light";break;
		case 11: result = "fire hydrant";break;
		case 12: result = "stop sign";break;
		case 13: result = "parking meter";break;
		case 14: result = "bench";break;
		case 15: result = "bird";break;
		case 16: result = "cat";break;
		case 17: result = "dog";break;
		case 18: result = "horse";break;
		case 19: result = "sheep";break;
		case 20: result = "cow";break;
		case 21: result = "elephant";break;
		case 22: result = "bear";break;
		case 23: result = "zebra";break;
		case 24: result = "giraffe";break;
		case 25: result = "backpack";break;
		case 26: result = "umbrella";break;
		case 27: result = "handbag";break;
		case 28: result = "tie";break;
		case 29: result = "suitcase";break;
		case 30: result = "frisbee";break;
		case 31: result = "skis";break;
		case 32: result = "snowboard";break;
		case 33: result = "sports ball";break;
		case 34: result = "kite";break;
		case 35: result = "baseball bat";break;
		case 36: result = "baseball glove";break;;
		case 37: result = "skateboard";break;
		case 38: result = "surfboard";break;
		case 39: result = "tennis racket";break;
		case 40: result = "bottle";break;
		case 41: result = "wine glass";break;
		case 42: result = "cup";break;
		case 43: result = "fork";break;
		case 44: result = "knife";break;
		case 45: result = "spoon";break;
		case 46: result = "bowl";break;
		case 47: result = "banana";break;
		case 48: result = "apple";break;
		case 49: result = "sandwich";break;
		case 50: result = "orange";break;
		case 51: result = "broccoli";break;
		case 52: result = "carrot";break;
		case 53: result = "hot dog";break;
		case 54: result = "pizza";break;
		case 55: result = "donut";break;
		case 56: result = "cake";break;
		case 57: result = "chair";break;
		case 58: result = "couch";break;
		case 59: result = "potted plant";break;
		case 60: result = "bed";break;
		case 61: result = "dining table";break;
		case 62: result = "toilet";break;
		case 63: result = "tv";break;
		case 64: result = "laptop";break;
		case 65: result = "mouse";break;
		case 66: result = "remote";break;
		case 67: result = "keyboard";break;
		case 68: result = "cell phone";break;
		case 69: result = "microwave";break;
		case 70: result = "oven";break;
		case 71: result = "toaster";break;
		case 72: result = "sink";break;
		case 73: result = "refrigerator";break;
		case 74: result = "book";break;
		case 75: result = "clock";break;
		case 76: result = "vase";break;
		case 77: result = "scissors";break;
		case 78: result = "teddy bear";break;
		case 79: result = "hair drier";break;
		case 80: result = "toothbrush";break;
    }
	return result;
}

#endif /* _EXAMPLE_COMMON_H_ */
