/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
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
