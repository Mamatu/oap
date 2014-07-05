/* 
 * File:   Error.cpp
 * Author: marcin
 * 
 * Created on 09 December 2012, 14:39
 */

#include "WrapperInterfaces.h"
#include "Argument.h"
#include "Buffer.h"
#include <algorithm>
#include "ArrayTools.h"
namespace utils {

    Identificator::Identificator(OglaObject* root) : Leaf<OglaObject>(root) {
    }

    void Identificator::setName(const std::string& name) {
        this->name = (name);
    }

    void Identificator::setName(const char* name) {
        this->name = (std::string(name));
    }

    Identificator::Identificator(const char* _name, OglaObject* root) : Leaf<OglaObject>(root) {
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

    OglaFunction::FunctionsList OglaFunction::functionsList;

    OglaFunction::FunctionsList::FunctionsList() {
    }

    LHandle OglaFunction::RegisterCallback(Callback_f callback, void* ptr) {
        OglaFunction::functionsList.functionsMutex.lock();
        LHandle id = OglaFunction::functionsList.callbacks.add(callback, ptr);
        OglaFunction::functionsList.functionsMutex.unlock();
        return id;
    }

    void OglaFunction::UnegisterCallback(LHandle callbackID) {
        OglaFunction::functionsList.functionsMutex.lock();
        OglaFunction::functionsList.callbacks.remove(callbackID);
        OglaFunction::functionsList.functionsMutex.unlock();
    }

    int OglaFunction::EVENT_CREATE_FUNCTION = 0;
    int OglaFunction::EVENT_DESTROY_FUNCTION = 1;

    OglaFunction::OglaFunction(const char* _name, const utils::ArgumentType* _inputArgs, int _inputArgc, OglaObject* _object,
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

        OglaFunction::functionsList.functionsMutex.lock();
        OglaFunction::functionsList.functions.push_back(this);
        OglaFunction::functionsList.callbacks.invoke(OglaFunction::EVENT_CREATE_FUNCTION, this);
        OglaFunction::functionsList.functionsMutex.unlock();
    }

    OglaFunction::OglaFunction(const char* _name, const utils::ArgumentType* _inputArgs, int _inputArgc, const utils::ArgumentType* _outputArgs, int _outputArgc) :
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
        OglaFunction::functionsList.functionsMutex.lock();
        OglaFunction::functionsList.functions.push_back(this);
        OglaFunction::functionsList.callbacks.invoke(OglaFunction::EVENT_CREATE_FUNCTION, this);
        OglaFunction::functionsList.functionsMutex.unlock();
    }

    OglaFunction::~OglaFunction() {
        if (inputArguments) {
            delete[] inputArguments;
        }
        if (outputArguments) {
            delete[] outputArguments;
        }
        OglaFunction::functionsList.functionsMutex.lock();
        std::vector<OglaFunction*>::iterator it = std::find(OglaFunction::functionsList.functions.begin(), OglaFunction::functionsList.functions.end(), this);
        OglaFunction::functionsList.functions.erase(it);
        OglaFunction::functionsList.callbacks.invoke(OglaFunction::EVENT_DESTROY_FUNCTION, this);
        OglaFunction::functionsList.functionsMutex.unlock();
    }

    void OglaFunction::invoke(utils::Writer& input, utils::Reader& output) {
        Writer writer;
        Reader reader;
        reader.setBuffer(&input);
        this->invoked(reader, &writer);
        output.setBuffer(&writer);
    }

    void OglaFunction::invoke(utils::Reader& input, utils::Writer& output) {
        this->invoked(input, &output);
    }

    void OglaFunction::invoke(utils::Reader& input, utils::Reader& output) {
        Writer writer;
        this->invoked(input, &writer);
        output.setBuffer(&writer);
    }

    void OglaFunction::invoke(utils::Writer& input, utils::Writer& output) {
        Reader reader;
        reader.setBuffer(&input);
        this->invoked(reader, &output);
    }

    void OglaFunction::invoke(utils::Writer& input) {
        Reader reader;
        reader.setBuffer(&input);
        this->invoked(reader, NULL);
    }

    void OglaFunction::invoke(utils::Reader& input) {
        this->invoked(input, NULL);
    }

    int OglaFunction::getInputArgc() const {
        return inputArgc;
    }

    ArgumentType OglaFunction::getInputArgumentType(int index) const {
        if (index > inputArgc) {
            return ARGUMENT_TYPE_INVALID;
        }
        return inputArguments[index];
    }

    const ArgumentType* OglaFunction::getInputArgumentsTypes() const {
        return inputArguments;
    }

    int OglaFunction::getOutputArgc() const {
        return outputArgc;
    }

    ArgumentType OglaFunction::getOutputArgumentType(int index) const {
        if (index > outputArgc) {
            return ARGUMENT_TYPE_INVALID;
        }
        return outputArguments[index];
    }

    const ArgumentType* OglaFunction::getOutputArgumentsTypes() const {
        return outputArguments;
    }

    bool OglaFunction::equals(OglaFunction& function) const {
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

    bool OglaFunction::equals(const char* name, const utils::ArgumentType* inputArguments, int inputArgc) const {
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
            OglaObject* root, const utils::ArgumentType* _outputArgs, int _outputArgc, void* _ptr) :
    OglaFunction(_name, _inputArgs, _inputArgc, root, _outputArgs, _outputArgc), functionPtr(_functionPtr), function(NULL), ptr(_ptr) {
    }

    FunctionProxy::FunctionProxy(OglaFunction* _function) :
    OglaFunction(_function->getName(), _function->getInputArgumentsTypes(), _function->getInputArgc(),
    _function->getRoot(), _function->getOutputArgumentsTypes(), _function->getOutputArgc()), functionPtr(NULL), function(_function) {
    }

    FunctionProxy::FunctionProxy(OglaFunction* _function, const char* name) :
    OglaFunction(name, _function->getInputArgumentsTypes(), _function->getInputArgc(),
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

    OglaObject::OglaObject(const char* name, OglaObject* root) : Identificator(name, root), objectsChain(false), functions(NULL), objects(NULL) {
    }

    OglaObject* OglaObject::CreateObjects(const char** names, int namesCount) {
        OglaObject* root = NULL;
        for (int fa = 0; fa < namesCount; fa++) {
            OglaObject* object = new OglaObject(names[fa], root);
            root = object;
        }
        return root;
    }

    OglaObject::OglaObject(const char** names, int namesCount) : Identificator(names[namesCount - 1]), objectsChain(true) {
        OglaObject* root = OglaObject::CreateObjects(names, namesCount - 1);
        this->setRoot(root);
    }

    void OglaObject::DestroyObjects(OglaObject* object) {
        OglaObject* root = object->getRoot();
        if (root != NULL) {
            OglaObject::DestroyObjects(root);
        }
        delete object;
    }

    void OglaObject::ConvertToStrings(OglaObject* object, const char*** names, int& namesCount) {
        if (object) {
            OglaObject* root = object->getRoot();
            if (root) {
                OglaObject::ConvertToStrings(root, names, namesCount);
                ArrayTools::add(names, namesCount, object->getName());
            } else {
                ArrayTools::add(names, namesCount, object->getName());
            }
        }
    }

    void OglaObject::ConvertToStrings(OglaObject* object, std::vector<std::string>& names) {
        if (object) {
            OglaObject* root = object->getRoot();
            if (root) {
                OglaObject::ConvertToStrings(root, names);
                names.push_back(object->getName());
            } else {
                names.push_back(object->getName());
            }
        }
    }

    void OglaObject::setObjectsContainer(Container<OglaObject>* objects) {
        this->objects = objects;
    }

    void OglaObject::setFunctionsContainer(Container<OglaFunction>* function) {
        this->functions = functions;
    }

    OglaObject::~OglaObject() {
    }

    Container<OglaObject>* OglaObject::getObjectsContainer() const {
        return this->objects;
    }

    Container<OglaFunction>* OglaObject::getFunctionsContainer() const {
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