/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
// General includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>
#include <fstream>

// Hailo includes
#include "common/math.hpp"
#include "common/hailo_objects.hpp"
#include "common/tensors.hpp"
#include "common/nms.hpp"
#include "common/labels/coco_eighty.hpp"
#include "postprocess.hpp"

using namespace xt::placeholders;


//******************************************************************
//  DEFAULT FILTER
//******************************************************************
void filter(HailoROIPtr roi)
{
    // Write the extracted feature vectors as rows of decimal numbers to output.txt
    // 
    std::ofstream debugFile("debug.txt", std::ios::app);
    std::ofstream outputFile("output.txt", std::ios::app);

    for (HailoTensorPtr elementPtr : roi->get_tensors()) {
        HailoTensor element = *elementPtr;
        uint8_t * data = element.data();
        uint32_t size = element.size();
 
        debugFile << "filter() got called\n";        
        
        if (outputFile.is_open()) {
            for (size_t i = 0; i < size; ++i) {
                int decimalValue = static_cast<int>(data[i]);
                outputFile << decimalValue << " ";
            }
            
        }        
        debugFile << "size: " << size << "\n";
    }
    debugFile << "----------\n\n";
    debugFile.close();
    outputFile << "\n";
    outputFile.close();
}
