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
