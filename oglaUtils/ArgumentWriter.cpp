/* 
 * File:   ArgumentWriter.cpp
 * Author: mmatula
 * 
 * Created on February 20, 2014, 7:18 PM
 */

#include <vector>

#include "ArgumentWriter.h"
#include "Writer.h"
#include "ArrayTools.h"

namespace utils {

    ArgumentsWriter::ArgumentsWriter() {
    }

    ArgumentsWriter::~ArgumentsWriter() {
    }

    bool ArgumentsWriter::write(const char* buffer, int size, std::type_info& typeInfo) {
        utils::ArgumentType argumentType = utils::ArgumentInfo::GetArgumentType(typeInfo);
        argumentsTypes.push_back(argumentType);
        Writer::write(buffer, size, typeInfo);
    }

    utils::ArgumentType* ArgumentsWriter::getArgumentsTypes(int& length) {
        utils::ArgumentType* output = new ArgumentType[argumentsTypes.size()];
        std::copy(argumentsTypes.begin(), argumentsTypes.end(), output);
        length = this->argumentsTypes.size();
        return output;
    }
}
