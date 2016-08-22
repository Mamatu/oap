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




#include <algorithm>
#include <vector>

#include "OapServer.h"
#include "Status.h"
#include "ObjectInfoImpl.h"

#define FUNCTION_CREATE_MODULE_ENTITY "CreateModuleEntity"
#define FUNCTION_DESTROY_MODULE_ENTITY "DestroyModuleEntity"
#define FUNCTION_START_PROCESS "StartProcess"
#define FUNCTION_STOP_PROCESS "StopProcess"
#define FUNCTION_SAVE_DATA "SaveData"
#define FUNCTION_LOAD_DATA "LoadData"
#define FUNCTION_EXECUTE_FUNCTION "ExecuteFunction"

namespace core {

class ObjectLoader : public Loader<utils::OapObject> {
    Loader<utils::OapObject>::Callback* callback;

public:
    utils::Callbacks callbacksManager;
    ObjectLoader(OapServer* thiz);
    virtual ~ObjectLoader();
};

ObjectLoader::ObjectLoader(OapServer* root) : Loader<utils::OapObject>("LoadObjects", "UnloadObject", "GetObjectsCount") {

    class CallbackImpl1 : public Loader<utils::OapObject>::Callback {
    public:
        uint count;
        OapServer* root;
        utils::Callbacks& callbacksManager;

        CallbackImpl1(OapServer* _root, utils::Callbacks& _callbacksManager) : count(0), root(_root), callbacksManager(_callbacksManager) {
        }

        bool setCount(uint count) {
            this->count = count;
            return true;
        }

        bool setLoadedImpls(utils::OapObject** impls, uint handle) {
            for (int fa = 0; fa<this->count; fa++) {
                impls[fa]->getFunctionsContainer();
            }
            return true;
        }

        void setUnloadedImpl(utils::OapObject* impl) {

        }
    };
    callback = new CallbackImpl1(root, this->callbacksManager);
    this->callbacks.push_back(callback);
}

ObjectLoader::~ObjectLoader() {
}

class ModuleHandle {
public:

    ModuleHandle() : stoped(true) {
    }

    ~ModuleHandle() {
    }

    std::vector<std::string> strings;
    uint objecthandle;
    uint processApiHandle;
    bool stoped;
};

int OapServer::registerModules(const char* path, LHandle& handle) {
    return this->moduleLoader->load(path, handle);
}

int OapServer::unregisterModule(LHandle handle) {
    return this->moduleLoader->remove(handle);
}

void OapServer::ModuleCallback(int event, void* object, void* userPtr) {
    if (event == 0) {
    }
}

void OapServer::RegisterObject(utils::Reader& input, utils::Writer* output, void* ptr) {
    OapServer* thiz = (OapServer*) ptr;
    std::string url = "";
    input.read(url);
    LHandle handle = 0;
    //uint callbackHandle = thiz->moduleLoader->callbacksManager.add(OapServer::ModuleCallback, &modules);
    int status = thiz->registerModules(url.c_str(), handle);
    int convertedHandle = *reinterpret_cast<int*> (&handle);
    //thiz->moduleLoader->callbacksManager.remove(callbackHandle);
    output->write(status);

}

void OapServer::UnregisterObject(utils::Reader& input, utils::Writer* output, void* ptr) {
    OapServer* thiz = (OapServer*) ptr;
    int convertedHandle = input.readInt();
    LHandle handle = *reinterpret_cast<LHandle*> (&convertedHandle);
    thiz->unregisterModule(handle);
}

void OapServer::Execute(void* ptr) {
    OapServer* oapServer = (OapServer*) ptr;
    while (oapServer->destroyed == false) {
        LHandle connectionID = oapServer->rpc->waitOnConnection();
        //oapServer->invokeCallbacks(OapServer::EVENT_REMOTE_CONNECTION, &connectionID);
    }
}

OapServer::OapServer(int port, bool registerDefaultMethods) : rpc(new utils::RpcImpl("", port, this)),
    moduleLoader(new ObjectLoader(this)), destroyed(false) {
    if (registerDefaultMethods) {
        utils::ArgumentType args1[] = {utils::ARGUMENT_TYPE_STRING};
        utils::ArgumentType args2[] = {utils::ARGUMENT_TYPE_INT, utils::ARGUMENT_TYPE_INT};
        utils::ArgumentType args3[] = {utils::ARGUMENT_TYPE_INT};
        utils::ArgumentType args4[] = {utils::ARGUMENT_TYPE_INT};
        rpc->registerCall("registerObject", OapServer::RegisterObject, args1, sizeof (args1) / sizeof (utils::ArgumentType), args2, sizeof (args2) / sizeof (utils::ArgumentType), this);
        rpc->registerCall("unregisterObject", OapServer::UnregisterObject, args3, sizeof (args3) / sizeof (utils::ArgumentType), args4, sizeof (args4) / sizeof (utils::ArgumentType), this);
    }
    mainThread.setFunction(OapServer::Execute, this);
    mainThread.run();
}

OapServer::~OapServer() {
    this->destroy();
    delete moduleLoader;
}

void OapServer::destroy() {
    this->destroyed = true;
    delete rpc;
    mainThread.yield();
    for (uint fa = 0; fa < miHandles.size(); fa++) {
        ModuleHandle* miHandle = reinterpret_cast<ModuleHandle*> (miHandles[fa]);
        delete miHandle;
    }
}

char* copyStr(const char* arg) {
    char* out = new char[strlen(arg) + 1];
    memcpy(out, arg, strlen(arg));
    out[strlen(arg)] = '\0';
    return out;
}

void OapServer::registerFunctions(utils::Container<utils::OapFunction>* functionsContainer) {
    if (functionsContainer) {
        uint count = functionsContainer->getCount();
        for (uint fa = 0; fa < count; fa++) {
            utils::OapFunction* function = functionsContainer->get(fa);
            this->registerFunction(function);
        }
    }
}

void OapServer::registerObjects(utils::Container<utils::OapObject>* objectsContainer) {
    if (objectsContainer) {
        uint count = objectsContainer->getCount();
        for (uint fa = 0; fa < count; fa++) {
            utils::OapObject* object = objectsContainer->get(fa);
            this->registerFunctions(object->getFunctionsContainer());
            this->registerObjects(object->getObjectsContainer());
            utils::ObjectInfoImpl objectInfoImpl(object);
            this->invokeCallbacks(OapServer::EVENT_REGISTER_OBJECT, &objectInfoImpl);
        }
    }
}

class ObjectHandle {
public:
    std::vector<uint> functionsHandles;
    std::vector<uint> objectsHandles;
    std::vector<utils::FunctionProxy*> functions;

    ~ObjectHandle() {
        for (uint fa = 0; fa < functions.size(); fa++) {
            delete functions[fa];
        }
        objectsHandles.clear();
        functionsHandles.clear();
    }
};

Status OapServer::registerFunction(utils::OapFunction* function) {
    if (function == NULL) {
        return STATUS_INVALID_ARGUMENT;
    }
    this->rpc->registerCall(function);
    return STATUS_OK;
}

void OapServer::wait() {
    this->mainThread.yield();
}

Status OapServer::executeFunction(LHandle connectionID, LHandle functionHandle, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
    this->rpc->call(connectionID, functionHandle, input, args, argc, output);
    return STATUS_OK;
}

Status OapServer::executeFunction(LHandle connectionID, const char* functionName, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
    if (functionName != NULL) {
        this->rpc->call(connectionID, functionName, input, args, argc, output);
        return STATUS_OK;
    }
    return STATUS_INVALID_ARGUMENT;
}

Status OapServer::executeFunction(LHandle connectionID, const char** functionNames, uint namesCount, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
    for (uint fa = 0; fa < namesCount; fa++) {
        if (functionNames[fa] == NULL) {
            return STATUS_INVALID_ARGUMENT;
        }
    }
    if (functionNames != NULL) {
        this->rpc->call(connectionID, functionNames, namesCount, input, args, argc, output);
        return STATUS_OK;
    }
    return STATUS_INVALID_ARGUMENT;
}

/*
void OapServer::CreateModuleEntity(utils::Reader& reader, utils::Writer* output, void* ptr) {
    OapServer* apiServer = (OapServer*) ptr;
    char* name = reader.getStr();
    uint handle = 0;
    Status code = apiServer->createProcess(name, handle);
    delete[] name;
    output->write(handle);
    output->write(code);
}

void OapServer::DestroyModuleEntity(utils::Reader& reader, utils::Writer* output, void* data_ptr) {
    OapServer* apiServer = (OapServer*) data_ptr;
    uint handle = reader.getInt();
    Status code = apiServer->destroyProcess(handle);
    output->write(code);
}

void OapServer::StartProcess(utils::Reader& reader, utils::Writer* output, void* data_ptr) {
    OapServer* apiServer = (OapServer*) data_ptr;
    uint handle = reader.getInt();
    Status code = apiServer->startProcess(handle);
    output->write(code);
}

void OapServer::StopProcess(utils::Reader& reader, utils::Writer* output, void* data_ptr) {
    OapServer* apiServer = (OapServer*) data_ptr;
    uint handle = reader.getInt();
    Status code = apiServer->stopProcess(handle);
    output->write(code);
}

bool OapServer::SearchImpl(const char** keys, const char** values, uint count, void* ptr) {
    std::pair<OapServer*, void*>* data1 = (std::pair<OapServer*, void*>*) ptr;
    std::pair<uint, uint>* data = (std::pair<uint, uint>*) data1->second;
    utils::Writer input;
    input.write(count);
    for (uint fa = 0; fa < count; fa++) {
        input.putStr(keys[fa]);
        input.putStr(values[fa]);
    }
    utils::ArgumentType args[] = {
        utils::ARGUMENT_TYPE_INT,
        utils::ARGUMENT_TYPE_ARRAY_STRINGS,
        utils::ARGUMENT_TYPE_ARRAY_STRINGS
    };
    utils::Reader output;
    data1->first->rpc->call(data->first, data->second, input, args, sizeof (args) / sizeof (int), output);
    return output.getBool();
}
     
void OapServer::LoadData(utils::Reader& input, utils::Writer* output, void* ptr) {
    OapServer* apiServer = (OapServer*) ptr;
    uint moduleInstanceHandle = input.getInt();
    uint functionHandle = input.getInt();
    uint connectionID = (uint) input.getInt();

    std::pair<uint, uint> data(connectionID, functionHandle);
    std::pair<OapServer*, void*> data1(apiServer, &data);

    Status code = apiServer->loadData(moduleInstanceHandle, OapServer::SearchImpl, &data1);
    output->write(code);
}

void OapServer::SaveData(utils::Reader& input, utils::Writer* output, void* ptr) {
    OapServer* apiServer = (OapServer*) ptr;
    uint moduleInstanceHandle = input.getInt();
    uint count = (uint) input.getInt();
    const char** keys = new const char*[count];
    const char** values = new const char*[count];
    for (uint fa = 0; fa < count; fa++) {
        keys[fa] = input.getStr();
        values[fa] = input.getStr();
    }
    Status code = apiServer->saveData(moduleInstanceHandle, keys, values, count);
    for (uint fa = 0; fa < count; fa++) {
        delete[] keys[fa];
        delete[] values[fa];
    }
    output->write(code);
}
 */
int OapServer::EVENT_REGISTER_FUNCTION = 0;
int OapServer::EVENT_UNREGISTER_FUNCTION = 1;
int OapServer::EVENT_REMOTE_CONNECTION = 2;
int OapServer::EVENT_REGISTER_OBJECT = 3;
int OapServer::EVENT_UNREGISTER_OBJECT = 4;
}
