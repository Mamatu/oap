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




#include "WrapperInterfaces.h"
#include "Argument.h"
#include "Buffer.h"
#include <algorithm>
#include "ArrayTools.h"
namespace utils {

    Identificator::Identificator(OapObject* root) : Leaf<OapObject>(root) {
    }

    void Identificator::setName(const std::string& name) {
        this->name = (name);
    }

    void Identificator::setName(const char* name) {
        this->name = (std::string(name));
    }

    Identificator::Identificator(const char* _name, OapObject* root) : Leaf<OapObject>(root) {
        this->name = std::string(_name);
    }

    Identificator::~Identificator() {
        this->name.clear();
    }

    const char* Identificator::getName() const {
        return this->name.c_str();
    }

    bool Identificator::equals(Identificator& another) const {
        return another.name == this->name;
    }

    bool Identificator::equals(const char* name) const {
        return std::string(name) == this->name;
    }

    Info::Info() : Identificator() {
    }

    Info::Info(const char* name) : Identificator(name) {
    }

    Info::~Info() {
    }

    OapFunction::FunctionsList OapFunction::functionsList;

    OapFunction::FunctionsList::FunctionsList() {
    }

    LHandle OapFunction::RegisterCallback(Callback_f callback, void* ptr) {
        OapFunction::functionsList.functionsMutex.lock();
        LHandle id = OapFunction::functionsList.callbacks.add(callback, ptr);
        OapFunction::functionsList.functionsMutex.unlock();
        return id;
    }

    void OapFunction::UnegisterCallback(LHandle callbackID) {
        OapFunction::functionsList.functionsMutex.lock();
        OapFunction::functionsList.callbacks.remove(callbackID);
        OapFunction::functionsList.functionsMutex.unlock();
    }

    int OapFunction::EVENT_CREATE_FUNCTION = 0;
    int OapFunction::EVENT_DESTROY_FUNCTION = 1;

    OapFunction::OapFunction(const char* _name, const utils::ArgumentType* _inputArgs, int _inputArgc, OapObject* _object,
            const utils::ArgumentType* _outputArgs, int _outputArgc) : Identificator(_name, _object),
    inputArguments(NULL), outputArguments(NULL), inputArgc(0), outputArgc(0) {
        if (_inputArgs) {
            this->inputArguments = new utils::ArgumentType[_inputArgc];
            memcpy(this->inputArguments, _inputArgs, _inputArgc);
            this->inputArgc = _inputArgc;
        }
        if (_outputArgs) {
            this->outputArguments = new utils::ArgumentType[_outputArgc];
            memcpy(this->outputArguments, _outputArgs, _outputArgc);
            this->outputArgc = _outputArgc;
        }

        OapFunction::functionsList.functionsMutex.lock();
        OapFunction::functionsList.functions.push_back(this);
        OapFunction::functionsList.callbacks.invoke(OapFunction::EVENT_CREATE_FUNCTION, this);
        OapFunction::functionsList.functionsMutex.unlock();
    }

    OapFunction::OapFunction(const char* _name, const utils::ArgumentType* _inputArgs, int _inputArgc, const utils::ArgumentType* _outputArgs, int _outputArgc) :
    Identificator(_name, NULL),
    inputArguments(NULL), outputArguments(NULL), inputArgc(0), outputArgc(0) {
        if (_inputArgs) {
            this->inputArguments = new utils::ArgumentType[_inputArgc];
            memcpy(this->inputArguments, _inputArgs, _inputArgc);
            this->inputArgc = _inputArgc;
        }
        if (_outputArgs) {
            this->outputArguments = new utils::ArgumentType[_outputArgc];
            memcpy(this->outputArguments, _outputArgs, _outputArgc);
            this->outputArgc = _outputArgc;
        }
        OapFunction::functionsList.functionsMutex.lock();
        OapFunction::functionsList.functions.push_back(this);
        OapFunction::functionsList.callbacks.invoke(OapFunction::EVENT_CREATE_FUNCTION, this);
        OapFunction::functionsList.functionsMutex.unlock();
    }

    OapFunction::~OapFunction() {
        if (inputArguments) {
            delete[] inputArguments;
        }
        if (outputArguments) {
            delete[] outputArguments;
        }
        OapFunction::functionsList.functionsMutex.lock();
        std::vector<OapFunction*>::iterator it = std::find(OapFunction::functionsList.functions.begin(), OapFunction::functionsList.functions.end(), this);
        OapFunction::functionsList.functions.erase(it);
        OapFunction::functionsList.callbacks.invoke(OapFunction::EVENT_DESTROY_FUNCTION, this);
        OapFunction::functionsList.functionsMutex.unlock();
    }

    void OapFunction::invoke(utils::Writer& input, utils::Reader& output) {
        Writer writer;
        Reader reader;
        reader.setBuffer(&input);
        this->invoked(reader, &writer);
        output.setBuffer(&writer);
    }

    void OapFunction::invoke(utils::Reader& input, utils::Writer& output) {
        this->invoked(input, &output);
    }

    void OapFunction::invoke(utils::Reader& input, utils::Reader& output) {
        Writer writer;
        this->invoked(input, &writer);
        output.setBuffer(&writer);
    }

    void OapFunction::invoke(utils::Writer& input, utils::Writer& output) {
        Reader reader;
        reader.setBuffer(&input);
        this->invoked(reader, &output);
    }

    void OapFunction::invoke(utils::Writer& input) {
        Reader reader;
        reader.setBuffer(&input);
        this->invoked(reader, NULL);
    }

    void OapFunction::invoke(utils::Reader& input) {
        this->invoked(input, NULL);
    }

    int OapFunction::getInputArgc() const {
        return inputArgc;
    }

    ArgumentType OapFunction::getInputArgumentType(int index) const {
        if (index > inputArgc) {
            return ARGUMENT_TYPE_INVALID;
        }
        return inputArguments[index];
    }

    const ArgumentType* OapFunction::getInputArgumentsTypes() const {
        return inputArguments;
    }

    int OapFunction::getOutputArgc() const {
        return outputArgc;
    }

    ArgumentType OapFunction::getOutputArgumentType(int index) const {
        if (index > outputArgc) {
            return ARGUMENT_TYPE_INVALID;
        }
        return outputArguments[index];
    }

    const ArgumentType* OapFunction::getOutputArgumentsTypes() const {
        return outputArguments;
    }

    bool OapFunction::equals(OapFunction& function) const {
        if (Identificator::equals(function)) {
            for (int fa = 0; fa < inputArgc; fa++) {
                if (function.inputArguments[fa] != this->inputArguments[fa]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    bool OapFunction::equals(const char* name, const utils::ArgumentType* inputArguments, int inputArgc) const {
        if (Identificator::equals(name)) {
            for (int fa = 0; fa < inputArgc; fa++) {
                if (inputArguments[fa] != this->inputArguments[fa]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    FunctionProxy::FunctionProxy(Function_f _functionPtr, const char* _name, const utils::ArgumentType* _inputArgs, int _inputArgc,
            OapObject* root, const utils::ArgumentType* _outputArgs, int _outputArgc, void* _ptr) :
    OapFunction(_name, _inputArgs, _inputArgc, root, _outputArgs, _outputArgc), functionPtr(_functionPtr), function(NULL), ptr(_ptr) {
    }

    FunctionProxy::FunctionProxy(OapFunction* _function) :
    OapFunction(_function->getName(), _function->getInputArgumentsTypes(), _function->getInputArgc(),
    _function->getRoot(), _function->getOutputArgumentsTypes(), _function->getOutputArgc()), functionPtr(NULL), function(_function) {
    }

    FunctionProxy::FunctionProxy(OapFunction* _function, const char* name) :
    OapFunction(name, _function->getInputArgumentsTypes(), _function->getInputArgc(),
    _function->getRoot(), _function->getOutputArgumentsTypes(), _function->getOutputArgc()), functionPtr(NULL), function(_function) {
    }

    FunctionProxy::~FunctionProxy() {
    }

    void FunctionProxy::invoked(utils::Reader& input, utils::Writer* output) {
        if (this->functionPtr) {
            this->functionPtr(input, output, this->ptr);
        } else if (this->function) {
            if (output) {
                this->function->invoke(input, *output);
            } else {
                this->function->invoke(input);
            }
        }
    }

    OapObject::OapObject(const char* name, OapObject* root) : Identificator(name, root), objectsChain(false), functions(NULL), objects(NULL) {
    }

    OapObject* OapObject::CreateObjects(const char** names, int namesCount) {
        OapObject* root = NULL;
        for (int fa = 0; fa < namesCount; fa++) {
            OapObject* object = new OapObject(names[fa], root);
            root = object;
        }
        return root;
    }

    OapObject::OapObject(const char** names, int namesCount) : Identificator(names[namesCount - 1]), objectsChain(true) {
        OapObject* root = OapObject::CreateObjects(names, namesCount - 1);
        this->setRoot(root);
    }

    void OapObject::DestroyObjects(OapObject* object) {
        OapObject* root = object->getRoot();
        if (root != NULL) {
            OapObject::DestroyObjects(root);
        }
        delete object;
    }

    void OapObject::ConvertToStrings(OapObject* object, const char*** names, int& namesCount) {
        if (object) {
            OapObject* root = object->getRoot();
            if (root) {
                OapObject::ConvertToStrings(root, names, namesCount);
                ArrayTools::add(names, namesCount, object->getName());
            } else {
                ArrayTools::add(names, namesCount, object->getName());
            }
        }
    }

    void OapObject::ConvertToStrings(OapObject* object, std::vector<std::string>& names) {
        if (object) {
            OapObject* root = object->getRoot();
            if (root) {
                OapObject::ConvertToStrings(root, names);
                names.push_back(object->getName());
            } else {
                names.push_back(object->getName());
            }
        }
    }

    void OapObject::setObjectsContainer(Container<OapObject>* objects) {
        this->objects = objects;
    }

    void OapObject::setFunctionsContainer(Container<OapFunction>* function) {
        this->functions = functions;
    }

    OapObject::~OapObject() {
    }

    Container<OapObject>* OapObject::getObjectsContainer() const {
        return this->objects;
    }

    Container<OapFunction>* OapObject::getFunctionsContainer() const {
        return this->functions;
    }

    FunctionsContainer::FunctionsContainer() {
    }

    FunctionsContainer::~FunctionsContainer() {
    }

    ObjectsContainer::ObjectsContainer() {
    }

    ObjectsContainer::~ObjectsContainer() {
    }

    DefaultFunctionsContainer::~DefaultFunctionsContainer() {
    }

}
