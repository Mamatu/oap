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




#ifndef OAP_ARGUMENT_H
#define	OAP_ARGUMENT_H

#include <string>
#include <typeinfo> 

namespace utils {

    enum ArgumentType {
        ARGUMENT_TYPE_INVALID = -1,
        ARGUMENT_TYPE_INT,
        ARGUMENT_TYPE_LONG,
        ARGUMENT_TYPE_FLOAT,
        ARGUMENT_TYPE_DOUBLE,
        ARGUMENT_TYPE_CHAR,
        ARGUMENT_TYPE_STRING,
        ARGUMENT_TYPE_BOOL,
        ARGUMENT_TYPE_BYTE,
        ARGUMENT_TYPE_OBJECT,
        ARGUMENT_TYPE_FUNCTION,
        ARGUMENT_TYPE_ARRAY_INTS,
        ARGUMENT_TYPE_ARRAY_LONGS,
        ARGUMENT_TYPE_ARRAY_FLOATS,
        ARGUMENT_TYPE_ARRAY_DOUBLES,
        ARGUMENT_TYPE_ARRAY_CHARS,
        ARGUMENT_TYPE_ARRAY_STRINGS,
        ARGUMENT_TYPE_ARRAY_BOOLS,
        ARGUMENT_TYPE_ARRAY_BYTES,
        ARGUMENT_TYPE_ARRAY_OBJECTS,
        ARGUMENT_TYPE_ARRAY_FUNCTIONS
    };

    class ArgumentInfo {
        static ArgumentType types[];
        static std::string typesStr[];
//        static std::pair<ArgumentType, ArgumentType> conversionPairs[];
        static int find(const std::string& type);
    public:
//        static bool IsPossibleConversion(ArgumentType from, ArgumentType to);

        template<typename T> static ArgumentType GetArgumentType(T t) {
            return GetArgumentType(typeid (t));
        }
        static ArgumentType GetArgumentType(std::type_info& typeInfo);
    };
}
#endif	/* ARGUMENT_H */
