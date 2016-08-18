
#ifndef ARGUMENTWRITER_H
#define	ARGUMENTWRITER_H

#include "Argument.h"
#include "Writer.h"


namespace utils {

    class ArgumentsWriter : public utils::Writer {
    public:
        ArgumentsWriter();
        virtual ~ArgumentsWriter();
        utils::ArgumentType* getArgumentsTypes(int& length);
    protected:
        virtual bool write(const char* buffer, int size, std::type_info& typeInfo);
    private:
        std::vector<utils::ArgumentType> argumentsTypes;
    };
}
#endif	/* ARGUMENTWRITER_H */

