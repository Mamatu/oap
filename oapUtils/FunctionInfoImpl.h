
#ifndef FUNCTIONEVENTIMPL_H
#define	FUNCTIONEVENTIMPL_H

#include "FunctionInfo.h"

namespace utils {

    class FunctionInfoImpl : public utils::FunctionInfo {
        uint namesCount;
        std::vector<std::string> names;
        utils::ArgumentType* inArgs;
        utils::ArgumentType* outArgs;
        uint inArgc;
        uint outArgc;
        bool deserialized;

    public:
        LHandle connectionID;
        LHandle functionID;

        FunctionInfoImpl();
        FunctionInfoImpl(utils::OapFunction* function);
        FunctionInfoImpl(const FunctionInfoImpl& event);
        FunctionInfoImpl(const char* _name, uint _connectionID,
                utils::ArgumentType* _inArgs, uint _inArgc, utils::ArgumentType* _outArgs, uint _outArgc);

        virtual ~FunctionInfoImpl();

        LHandle getFunctionHandle() const;
        LHandle getConnectionID() const;
        void getName(char*& name,uint index) const;
        int getNamesCount() const;
        utils::ArgumentType getInputArgumentType(uint index) const;
        uint getInputArgc() const;
        utils::ArgumentType getOutputArgumentType(uint index) const;
        uint getOutputArgc() const;
        void serialize(utils::Writer& setter);
        void deserialize(utils::Reader& Reader);
    };
}
#endif	/* FUNCTIONEVENTIMPL_H */

