/* 
 * File:   Argument.cpp
 * Author: marcin
 * 
 * Created on 03 January 2013, 22:56
 */

#include "Argument.h"
#include "ArrayTools.h"
#include "Writer.h"
#include "Reader.h"

#define GET_NAME(t) typeid(t).name()

namespace utils {

    ArgumentType ArgumentInfo::types[] = {ARGUMENT_TYPE_INT, ARGUMENT_TYPE_LONG, ARGUMENT_TYPE_FLOAT,
        ARGUMENT_TYPE_DOUBLE, ARGUMENT_TYPE_CHAR, ARGUMENT_TYPE_STRING, ARGUMENT_TYPE_BOOL, ARGUMENT_TYPE_BYTE,
        ARGUMENT_TYPE_OBJECT, ARGUMENT_TYPE_FUNCTION, ARGUMENT_TYPE_ARRAY_INTS, ARGUMENT_TYPE_ARRAY_LONGS,
        ARGUMENT_TYPE_ARRAY_FLOATS, ARGUMENT_TYPE_ARRAY_DOUBLES, ARGUMENT_TYPE_ARRAY_CHARS, ARGUMENT_TYPE_ARRAY_STRINGS, ARGUMENT_TYPE_ARRAY_BOOLS,
        ARGUMENT_TYPE_ARRAY_BYTES, ARGUMENT_TYPE_ARRAY_OBJECTS, ARGUMENT_TYPE_ARRAY_FUNCTIONS};

    std::string ArgumentInfo::typesStr[] = {GET_NAME(int), GET_NAME(long long), GET_NAME(float),
        GET_NAME(double), GET_NAME(char), GET_NAME(std::string), GET_NAME(bool), GET_NAME(char), "oap_serialized_object_type", "oap_function", ""};

/*    std::pair<ArgumentType, ArgumentType> ArgumentInfo::conversionPairs = {
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_FLOAT, ARGUMENT_TYPE_DOUBLE),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_ARRAY_FLOATS, ARGUMENT_TYPE_ARRAY_DOUBLES),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_DOUBLE, ARGUMENT_TYPE_FLOAT),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_ARRAY_DOUBLES, ARGUMENT_TYPE_ARRAY_FLOATS),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_INT, ARGUMENT_TYPE_LONG),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_LONG, ARGUMENT_TYPE_INT),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_BYTE, ARGUMENT_TYPE_CHAR),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_CHAR, ARGUMENT_TYPE_BYTE),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_STRING, ARGUMENT_TYPE_ARRAY_CHARS),
        std::pair<ArgumentType, ArgumentType>(ARGUMENT_TYPE_ARRAY_CHARS, ARGUMENT_TYPE_STRING)
    };

    bool ArgumentInfo::IsPossibleConversion(ArgumentType from, ArgumentType to) {

    }
*/
    int ArgumentInfo::find(const std::string& type) {
        int index = -1;
        for (int fa = 0; typesStr[fa] != ""; fa++) {
            if (type == typesStr[fa]) {
                index = fa;
                break;
            }
        }
        return index;
    }

    ArgumentType ArgumentInfo::GetArgumentType(std::type_info& typeInfo) {
        int begin = typeInfo.__is_pointer_p() == true ? 10 : 0;
        std::string name = typeInfo.name();
        if (begin != 0) {
            name = name.substr(0, name.size() - 1);
        }
        int index = find(name);
        ArgumentType argumentType = ARGUMENT_TYPE_INVALID;
        if (index > -1) {
            argumentType = (ArgumentType) (index);
        }
        return argumentType;
    }
}