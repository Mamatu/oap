/* 
 * File:   PythonApi.cpp
 * Author: mmatula
 * 
 * Created on September 11, 2013, 11:38 PM
 */

#include "PythonApi.h"
#include <map>
#include <python3.2/Python.h>

typedef std::map<PyObject*, void*> PyObjectsMap;
PyObjectsMap pyObjectsMap;
synchronization::Mutex pyObjectsMapMutex;
std::vector<char*> charPtrs;

static void AddImpl(PyObject* pyObject, void* ptr) {
    pyObjectsMapMutex.lock();
    pyObjectsMap[pyObject] = ptr;
    pyObjectsMapMutex.unlock();
}

static void* GetImpl(PyObject* pyObject) {
    pyObjectsMapMutex.lock();
    void* ptr = pyObjectsMap[pyObject];
    pyObjectsMapMutex.unlock();
    return ptr;
}

class ObjectType {
public:
    PyTypeObject pyTypeObject;
    std::string name;
};

static PyObject* py_server_load_module(PyObject *self, PyObject *args);
static PyObject* py_server_load_extension(PyObject *self, PyObject *args);
static PyObject* py_server_create_extension_entity(PyObject *self, PyObject *args);
static PyObject* py_server_create_module_entity(PyObject *self, PyObject *args);
static PyObject* py_server_execute_function(PyObject *self, PyObject *args);

static PyObject* py_client_load_module(PyObject *self, PyObject *args);
static PyObject* py_client_load_extension(PyObject *self, PyObject *args);
static PyObject* py_client_create_extension_entity(PyObject *self, PyObject *args);
static PyObject* py_client_create_module_entity(PyObject *self, PyObject *args);
static PyObject* py_client_execute_function(PyObject *self, PyObject *args);

struct IDObject {
    PyObject_HEAD
    uint id;
};

class ObjectEntity {
private:
    std::vector<ObjectEntity*> internalObjects;
    PyMethodDef* methods;
public:
    static PyObject* Function_f(PyObject* self, PyObject*args);
    static synchronization::Mutex functionMutex;
    PyObject* pyObject;
    static ObjectEntity* Create(utils::ObjectInfo* objectInfo);
    void setObject(utils::ObjectInfo* objectInfo);
    void setFunctions(utils::ObjectInfo* objectInfo);
};

synchronization::Mutex ObjectEntity::functionMutex;

PyObject* ObjectEntity::Function_f(PyObject* self, PyObject* args) {
    ObjectEntity::functionMutex.lock();
    
    ObjectEntity::functionMutex.unlock();
}

ObjectEntity* ObjectEntity::Create(utils::ObjectInfo* objectInfo) {
    ObjectEntity* objectEntity = new ObjectEntity();
    objectEntity->setObject(objectInfo);
    return objectEntity;
}

void ObjectEntity::setObject(utils::ObjectInfo* objectInfo) {
    this->setFunctions(objectInfo);
    for (uint fa = 0; fa < objectInfo->getObjectsCount(); fa++) {
        ObjectEntity* objectEntity = ObjectEntity::Create(objectInfo->getObjectInfo(fa));
        internalObjects.push_back(objectEntity);
    }
}

void ObjectEntity::setFunctions(utils::ObjectInfo* objectInfo) {
    this->methods = new PyMethodDef[objectInfo->getFunctionsCount() + 1];
    for (uint fa = 0; fa < objectInfo->getFunctionsCount(); fa++) {
        utils::FunctionInfo* fi = objectInfo->getFunctionInfo(fa);
        char* name = NULL;
        fi->getName(name);
        this->methods[fa] = {name, ObjectEntity::Function_f, METH_VARARGS, ""};
    }
    this->methods[objectInfo->getFunctionsCount()].ml_name = NULL;
    this->methods[objectInfo->getFunctionsCount()].ml_doc = NULL;
    this->methods[objectInfo->getFunctionsCount()].ml_meth = NULL;
    this->methods[objectInfo->getFunctionsCount()].ml_flags = 0;
}


PyMethodDef ApiServerMethods[] = {
    {"loadModule", py_server_load_module, METH_VARARGS, ""},
    {"createModuleEntity", py_server_load_extension, METH_VARARGS, ""},
    {"loadExtension", py_server_create_extension_entity, METH_VARARGS, ""},
    {"createExtensionEntity", py_server_create_module_entity, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


PyTypeObject ApiServerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "w.Server",
    sizeof (ApiServerObject),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    "Server objects",
};

PyTypeObject ApiClientType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "w.Client",
    sizeof (ApiClientObject),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    "Client objects",
};

static void CallbackImpl_f(int event, void* object, void* userPtr) {
    if (event == core::APIServer::EVENT_REGISTER_OBJECT) {
        ObjectType* objectType = new ObjectType();
        utils::ObjectInfo* objectInfo = (utils::ObjectInfo*)object;
        const char* namePtr = objectInfo->getName();
        objectType->name = std::string(namePtr);
        objectType->pyTypeObject.tp_name = objectType->name.c_str();

    }
}

static int py_server_tp_init(PyObject *self, PyObject *args) {
    int port = API_SERVER_DEFAULT_PORT;
    PyArg_ParseTuple(args, "i", &port);
    core::APIServer* apiServer = new core::APIServer(port);
    AddImpl(self, apiServer);
    return 1;
}

static void py_server_tp_dealloc(PyObject *self) {
    core::APIServer* apiServer = (core::APIServer*) pyObjectsMap[self];
    delete apiServer;
}

static PyObject* py_server_load_module(PyObject *self, PyObject *args) {
    char* path = NULL;
    PyArg_ParseTuple(args, "s", path);
    core::APIServer* apiServer = (core::APIServer*) GetImpl(self);
    uint id = apiServer->registerModules(path);
    IDObject* mobject = new IDObject();
    mobject->id = id;
    return (PyObject*) mobject;
}

static PyObject* py_server_load_extension(PyObject *self, PyObject *args) {
    char* path = NULL;
    PyObject* object = NULL;
    PyArg_ParseTuple(args, "O s", object, path);
    core::APIServer* apiServer = (core::APIServer*) GetImpl(self);
    IDObject* mobj = (IDObject*) object;
    uint id = apiServer->registerExtensions(mobj->id, path);
    IDObject* eobject = new IDObject();
    eobject->id = id;
    return eobject;
}

static PyObject* py_server_create_extension_entity(PyObject *self, PyObject *args) {
    PyObject* object = NULL;
    PyArg_ParseTuple(args, "O", object);
}

static PyObject* py_server_execute_function(PyObject *self, PyObject *args) {
    PyObject* object = NULL;
    char* functionName = NULL;
    char* args = NULL;
    PyArg_ParseTuple(args, "s s", &functionName, &args);

}

static int py_client_tp_init(PyObject *self, PyObject *args, PyObject *kwds) {
    int port = API_SERVER_DEFAULT_PORT;
    char* address = "localhost";

    if (PyArg_ParseTuple(args, "s i", address, &port) != 0) {
    }

    if (PyArg_ParseTuple(args, "i", &port) != 0) {
    }

    core::APIInterface* apiClient = client::newClient(address, port);
    apiClient->registerCallback(CallbackImpl_f, NULL);
    pyObjectsMap[self] = apiClient;
    return 1;
}

static void py_client_tp_dealloc(PyObject *self) {
    core::APIInterface* client = (core::APIInterface*) pyObjectsMap[self];
    delete client;
}

void Py_InitModule() {
    ApiServerType.tp_init = py_server_tp_init;
    ApiServerType.tp_dealloc = py_server_tp_dealloc;
    ApiClientType.tp_init = py_client_tp_init;
    ApiClientType.tp_dealloc = py_client_tp_dealloc;
}

void Py_DestroyModule() {
    for (uint fa = 0; fa < charPtrs.size(); fa++) {
        delete[] charPtrs[fa];
    }
}
