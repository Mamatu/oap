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




#include "FunctionInfoImpl.h"
namespace utils {

    FunctionInfoImpl::~FunctionInfoImpl() {
        if (this->deserialized) {
            delete[] inArgs;
            delete[] outArgs;
        }
    }

    FunctionInfoImpl::FunctionInfoImpl(utils::OapFunction* function) : deserialized(true) {
        this->outArgc = function->getOutputArgc();
        this->inArgc = function->getInputArgc();
        this->inArgs = this->inArgc <= 0 ? NULL : new utils::ArgumentType[this->inArgc];
        this->outArgs = this->outArgc <= 0 ? NULL : new utils::ArgumentType[this->outArgc];
        if (this->inArgc > 0) {
            memcpy(this->inArgs, function->getInputArgumentsTypes(), this->inArgc * sizeof (utils::ArgumentType));
        }
        if (this->outArgc > 0) {
            memcpy(this->outArgs, function->getOutputArgumentsTypes(), this->outArgc * sizeof (utils::ArgumentType));
        }
        OapObject::ConvertToStrings(function->getRoot(), this->names);
        this->names.push_back(std::string(function->getName()));
    }

    FunctionInfoImpl::FunctionInfoImpl(const FunctionInfoImpl& event) : deserialized(true) {
        this->connectionID = event.connectionID;
        this->outArgc = event.outArgc;
        this->inArgc = event.inArgc;
        this->outArgs = new utils::ArgumentType[this->outArgc];
        this->inArgs = new utils::ArgumentType[this->inArgc];
        memcpy(this->inArgs, event.inArgs, this->inArgc * sizeof (utils::ArgumentType));
        memcpy(this->outArgs, event.outArgs, this->outArgc * sizeof (utils::ArgumentType));
        std::copy(event.names.begin(), event.names.end(), this->names.begin());
    }

    FunctionInfoImpl::FunctionInfoImpl() :
    connectionID(0),
    inArgs(NULL),
    outArgs(NULL),
    inArgc(0),
    outArgc(0), deserialized(false) {
    }

    LHandle FunctionInfoImpl::getFunctionHandle() const {
        return functionID;
    }

    LHandle FunctionInfoImpl::getConnectionID() const {
        return this->connectionID;
    }

    void FunctionInfoImpl::getName(char*& name, uint index) const {
        const char* name1 = this->names[index].c_str();
        uint size = (strlen(name1) + 1) * sizeof (char);
        name = new char[size];
        memcpy(name, name1, size);
    }

    int FunctionInfoImpl::getNamesCount() const {
        return this->names.size();
    }

    utils::ArgumentType FunctionInfoImpl::getInputArgumentType(uint index) const {
        return this->inArgs[index];
    }

    uint FunctionInfoImpl::getInputArgc() const {
        return this->inArgc;
    }

    utils::ArgumentType FunctionInfoImpl::getOutputArgumentType(uint index) const {
        return this->outArgs[index];
    }

    uint FunctionInfoImpl::getOutputArgc() const {
        return this->outArgc;
    }

    void FunctionInfoImpl::serialize(Writer& setter) {
        setter.write(names.size());
        for (uint fa = 0; fa < names.size(); fa++) {
            setter.write(names[fa].c_str());
        }
        setter.write(inArgc);
        for (uint fa = 0; fa < inArgc; fa++) {
            setter.write(inArgs[fa]);
        }
        setter.write(outArgc);
        for (uint fa = 0; fa < outArgc; fa++) {
            setter.write(outArgs[fa]);
        }
        setter.write(this->connectionID);
    }

    void FunctionInfoImpl::deserialize(Reader& reader) {
        this->deserialized = true;
        int namesCount = reader.readInt();
        for (int fa = 0; fa < namesCount; fa++) {
            std::string name;
            reader.read(name);
            this->names.push_back(name);
        }
        inArgc = reader.readInt();
        for (uint fa = 0; fa < inArgc; fa++) {
            inArgs[fa] = (ArgumentType) reader.readInt();
        }
        outArgc = reader.readInt();
        for (uint fa = 0; fa < outArgc; fa++) {
            outArgs[fa] = (ArgumentType) reader.readInt();
        }
        this->connectionID = reader.readHandle();
    }
}
