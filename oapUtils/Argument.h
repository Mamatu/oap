
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

