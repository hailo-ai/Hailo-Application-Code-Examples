/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once

#include <iostream>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/schema.h"

namespace common
{
    /**
     * @brief validate that the json file (that its data is in stream) complies with the scehma rules.
     *
     * @param stream rapidjson::FileReadStream byte stream holding the json config file data
     * @param json_schema const char * holding the json schema rules
     * @return true in case the config file complies with the scehma rules.
     * @return false in case the config file doesn't comply with the scehma rules.
     */
    bool validate_json_with_schema(rapidjson::FileReadStream stream, const char *json_schema)
    {
        rapidjson::Document d;
        d.Parse(json_schema);
        rapidjson::SchemaDocument sd(d);
        rapidjson::SchemaValidator validator(sd);
        rapidjson::Document doc_config_json;
        rapidjson::Reader reader;
        if (!reader.Parse(stream, validator) && reader.GetParseErrorCode() != rapidjson::kParseErrorTermination)
        {
            // Schema validator error would cause kParseErrorTermination, which will handle it in next step.
            std::cerr << "JSON error (offset " << static_cast<unsigned>(reader.GetErrorOffset()) << "): " << GetParseError_En(reader.GetParseErrorCode()) << std::endl;
            throw std::runtime_error("Input is not a valid JSON");
        }

        // Check the validation result
        if (validator.IsValid())
        {
            return true;
        }
        else
        {
            std::cerr << "Input JSON is invalid" << std::endl;
            rapidjson::StringBuffer sb;
            validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
            std::cerr << "Invalid schema: " << sb.GetString() << std::endl;
            std::cerr << "Invalid keyword: " << validator.GetInvalidSchemaKeyword() << std::endl;
            sb.Clear();
            validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
            std::cerr << "Invalid document: " << sb.GetString() << std::endl;
            throw std::runtime_error("json config file doesn't follow schema rules");
        }
        return false;
    }
}