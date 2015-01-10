/* 
 * File:   RpcBase.cpp
 * Author: mmatula
 * 
 * Created on July 27, 2013, 10:35 PM
 */

#include "RpcBase.h"
#include "FunctionInfoImpl.h"
#include <algorithm>
namespace utils {

#define CONVERT_BUFFER_SIZE 256
const int RpcBase::EVENT_REGISTER_FUNCTION = 0;
const int RpcBase::EVENT_UNREGISTER_FUNCTION = 1;
const int RpcBase::EVENT_REMOTE_CONNECTION = 2;
const int RpcBase::MAGIC_NUMBER = 0xF98765;

RpcBase::~RpcBase() {
    delete server;
    SocketsWriters::iterator it;
    for (it = socketsWriters.begin(); it != socketsWriters.end(); it++) {
        if (it->second.buffer) {
            delete[] it->second.buffer;
        }
        if (it->second.writer) {
            delete it->second.writer;
        }
    }
    this->unregisterCallback(callbackHandle);
}

utils::Client* RpcBase::getSocket(LHandle clientID) {
    if (this->addresses.find(clientID) == this->addresses.end()) {
        return NULL;
    }
    return reinterpret_cast<utils::Client*> (clientID.getPtr());
}

RpcBase::RemoteConnectionEvent::RemoteConnectionEvent() {
}

RpcBase::RemoteConnectionEvent::~RemoteConnectionEvent() {
}

class RemoteConnectionEventImpl : public RpcBase::RemoteConnectionEvent {
protected:
    const char* address;
    uint16_t port;
    LHandle connectionID;

public:

    RemoteConnectionEventImpl(const char* _address, uint16_t _port, LHandle _connectionID) :
    address(_address), port(_port), connectionID(_connectionID) {
    }

    virtual ~RemoteConnectionEventImpl() {
    }

    void getAddress(char*& address) const {
        uint size = (strlen(this->address) + 1) * sizeof (char);
        address = new char[size];
        memcpy(address, this->address, size);
    }

    uint16_t getPort() const {
        return port;
    }

    LHandle getConnectionID() const {
        return connectionID;
    }
};

const int EXECUTE = 0;
const int RETURN = 1;
const int ESTABLISH_CONNECTION = 2;
const int SET_CLIENT_ID = 3;
const int GET_REGISTERED_FUNCTIONS = 4;
const int SET_REGISTERED_FUNCTIONS = 5;
const int WERE_REGISTERED_FUNCTIONS = 6;

int SetConnectionID = 1;
int SetActionID = 1 << 1;

RpcBase::Strings::Strings(const char* str) {
    this->stringsVec.push_back(std::string(str));
}

RpcBase::Strings::Strings(const char** strs, uint count) {
    for (uint fa = 0; fa < count; fa++) {
        this->stringsVec.push_back(std::string(strs[fa]));
    }
}

RpcBase::Strings::Strings(const std::vector<std::string>& _stringsVec) {
    for (uint fa = 0; fa < _stringsVec.size(); fa++) {
        this->stringsVec.push_back(_stringsVec[fa]);
    }
}

const char* RpcBase::Strings::c_str() {
    for (uint fa = 0; fa < stringsVec.size() - 1; fa++) {
        all += stringsVec[fa];
        all += "::";
    }
    all += stringsVec[stringsVec.size() - 1];
    return all.c_str();
}

bool RpcBase::Strings::operator<(const Strings& strings) const {
    if (this->stringsVec.size() != strings.stringsVec.size()) {
        return true;
    }
    for (uint fa = 0; fa < this->stringsVec.size(); fa++) {
        if (this->stringsVec[fa] < strings.stringsVec[fa]) {
            return true;
        }
    }
    return false;
}

void RpcBase::InputsDict::add(LHandle id, Serializable* serialize) {
    mutex.lock();
    serialize->serialize(*inputsMap[id]);
    mutex.unlock();
}

int RpcBase::InputsDict::count(LHandle id) {
    mutex.lock();
    int out = inputsMap.count(id);
    mutex.unlock();
    return out;
}

Reader* RpcBase::InputsDict::get(LHandle id) {
    Reader* reader = NULL;
    mutex.lock();
    if (inputsMap.count(id) > 0) {
        utils::Writer* writer = inputsMap[id];
        reader = new Reader();
        reader->setBuffer(writer);
    }
    mutex.unlock();
    return reader;
}

void RpcBase::InputsDict::remove(LHandle id) {
    mutex.lock();
    if (inputsMap.count(id) > 0) {
        inputsMap.erase(id);
    }
    mutex.unlock();
}

void RpcBase::executeFunction(Reader& reader, utils::Writer& writer, LHandle actionID, LHandle connectionID) {
    mutex.lock();
    uint wid = reader.readInt();
    OglaFunction* function = NULL;
    int rpcParameter = 0;
    if (wid == 0) {
        uint namesCount = 0;
        std::vector<std::string> strsVec;
        namesCount = reader.readInt();
        for (uint fa = 0; fa < namesCount; fa++) {
            char* functionName = reader.readStr();
            strsVec.push_back(std::string(functionName));
            delete[] functionName;
        }
        uint argc = 0;
        utils::ArgumentType* types = NULL;
        RpcBase::DeserializeArgumentsTypes(&types, argc, reader);
        bool b = false;
        Strings strings(strsVec);
        if (this->functionsMap.count(strings) > 0) {
            std::pair<FunctionsMap::iterator, FunctionsMap::iterator> it = this->functionsMap.equal_range(strings);
            for (; it.second != it.first && b == false; it.first++) {
                const char** names = new const char*[strings.stringsVec.size()];
                for (uint fa = 0; fa < strings.stringsVec.size(); fa++) {
                    names[fa] = strings.stringsVec[fa].c_str();
                }
                utils::OglaFunction* functionT = it.first->second.second;
                int rpcParameterT = it.first->second.first;
                if (functionT->equals(strings.stringsVec[strings.stringsVec.size() - 1].c_str(), types, argc)) {
                    function = functionT;
                    rpcParameter = rpcParameterT;
                }
                delete[] names;
            }
        } else {
            debug("Function: %s was not found. \n", strings.c_str());
        }
        if (types) {
            delete[] types;
        }
    } else {
        uint functionID = reader.readInt();
        OglaFunction* functionT = reinterpret_cast<OglaFunction*> (functionID);
        if (std::find(this->registeredFunctions.begin(), this->registeredFunctions.end(), functionT) != this->registeredFunctions.end()) {
            function = functionT;
        }
    }
    if (function) {
        Reader params;
        reader.read(&params);
        if (rpcParameter != 0) {
            uint size = params.getSize();
            char* buffer = new char[size + sizeof (uint)*2];
            params.getBufferBytes(buffer, 0, size);
            Writer writer1(buffer, size);
            if ((rpcParameter & SetActionID) != 0) {
                writer1.write(actionID);
            }
            if ((rpcParameter & SetConnectionID) != 0) {
                writer1.write(connectionID);
            }
            uint size1 = writer1.getSize();
            writer1.getBufferBytes(buffer, 0, size1);
            params.setBuffer(buffer, size1);
            delete[] buffer;
        }
        function->invoke(params, writer);
    }
    mutex.unlock();
}

void RpcBase::DeserializeArgumentsTypes(ArgumentType** types, uint& arguemntsNumber, Reader& reader) {
    if (types != NULL) {
        uint argc = reader.readInt();
        debug("argc == %u \n", argc);
        (*types) = new ArgumentType[argc];
        for (uint fa = 0; fa < argc; fa++) {
            (*types)[fa] = (ArgumentType) reader.readInt();
            debug("atype == %d\n", (*types)[fa]);
        }
        arguemntsNumber = argc;
    }
}

class FunctionArguments {
public:
    uint argc;
    ArgumentType* types;
};

class RpcBase::ServerImpl : public Server {
public:
    ServerImpl(int16_t port, RpcBase* rpcBase);
protected:
    void OnData(Socket* client, const char* buffer, int size);
private:
    RpcBase* m_rpcBase;
};

RpcBase::ServerImpl::ServerImpl(int16_t port, RpcBase* rpcBase) : Server(port),
m_rpcBase(rpcBase) {
}

void RpcBase::ServerImpl::OnData(Socket* client, const char* buffer, int size) {
    SocketsWriters::iterator it = m_rpcBase->socketsWriters.find(client);
    if (it == m_rpcBase->socketsWriters.end()) {
        Writer* w = new Writer();
        char* convertBuffer = new char[CONVERT_BUFFER_SIZE];
        RpcBase::SocketData socketData = {w, convertBuffer};
        m_rpcBase->socketsWriters[client] = socketData;
    }
    SocketData sd = m_rpcBase->socketsWriters[client];
    Writer* writer = sd.writer;
    char* tempBuffer = sd.buffer;
    writer->extendBuffer(buffer, size);
    if (writer->getSize() > 8) {
        Reader reader(*writer);
        while (1) {
            int magicNumber = reader.readInt();
            if (magicNumber == RpcBase::MAGIC_NUMBER) {
                uint sectionSize = reader.readInt();
                if (sectionSize <= reader.getSize() - reader.getPosition()) {
                    uint type = reader.readInt();
                    debug("type == %u \n", type);
                    RpcBase::ServerSubExecution(m_rpcBase, type, reader);

                    unsigned int len = writer->getSize() - (sectionSize + 8);
                    char* bytes = NULL;
                    if (len >= CONVERT_BUFFER_SIZE) {
                        bytes = new char[len];
                    } else {
                        bytes = tempBuffer;
                    }
                    writer->getBufferBytes(bytes, sectionSize + 8, len);
                    writer->setBuffer(bytes, len);
                    if (len >= CONVERT_BUFFER_SIZE) {
                        delete[] bytes;
                    }
                } else {
                    //writer->setBuffer(&reader);
                    break;
                }
            } else {
                break;
            }
        }
    }
    debug("Execution\n");
}

void RpcBase::ServerSubExecution(RpcBase* rpc, int type, Reader& reader) {
    LHandle actionID = reader.readHandle();
    LHandle clientID = reader.readHandle();
    debug("rpc == %p, actionID == %p, clientID == %p \n", rpc, actionID.getPtr(), clientID.getPtr());

    switch (type) {
        case EXECUTE:
            RpcBase::InvokeFunction(actionID, clientID, reader, rpc);
            break;
        case RETURN:
            RpcBase::Return(actionID, clientID, reader, rpc);
            break;
        case ESTABLISH_CONNECTION:
            RpcBase::EstablishConnection(actionID, clientID, reader, rpc);
            break;
        case SET_CLIENT_ID:
            RpcBase::SetMyClientID(actionID, clientID, reader, rpc);
            break;
        case GET_REGISTERED_FUNCTIONS:
            RpcBase::GetRegisteredFunctions(actionID, clientID, reader, rpc);
            break;
    };
    debug("Subexecution\n");
}

void RpcBase::getRegisteredFunctions(uint connectionID) {
}

void RpcBase::CallbackImpl(int eventType, void* eventObj, void* user_ptr) {
}

void RpcBase::putFunctionEvent(Writer& writer1, LHandle actionID, LHandle connectionID) {
    utils::Writer writer;
    writer.write(SET_REGISTERED_FUNCTIONS);
    writer.write(actionID);
    writer.write(this->addresses[connectionID]);
    this->callsMutex.lock();
    writer.write(registeredFunctions.size());
    for (uint fa = 0; fa < registeredFunctions.size(); fa++) {
        utils::OglaFunction* proxy = registeredFunctions[fa];
        FunctionInfoImpl info(proxy);
        info.connectionID = this->addresses[connectionID];
        writer.write(&info);
    }
    this->callsMutex.unlock();
    writer1.write(RpcBase::MAGIC_NUMBER);
    writer1.write(&writer);
}

void RpcBase::sendFunctionEvent(LHandle actionID, LHandle connectionID) {
    Writer writer;
    this->putFunctionEvent(writer, actionID, connectionID);
    utils::Client* socket = this->getSocket(connectionID);
    socket->send(&writer);
}

RpcBase::RpcBase(const char* _address, uint16_t _port, CallbacksManager* callbacksManager) : CallbacksManager(callbacksManager), address(_address),
port(_port), server(NULL), initID(0) {
    callbackHandle = this->registerCallback(CallbackImpl, this);
}

RpcBase::RpcBase(const char* _address, uint16_t _port, Callbacks* callbacks) : CallbacksManager(callbacks), address(_address),
port(_port), server(NULL), initID(0) {
    callbackHandle = this->registerCallback(CallbackImpl, this);
}

RpcBase::RpcBase(const char* _address, uint16_t _port) : CallbacksManager(), address(_address), port(_port), server(NULL), initID(0) {
    callbackHandle = this->registerCallback(CallbackImpl, this);
}

RpcBase::RpcBase(Callbacks* callbacks) : CallbacksManager(callbacks), address(""), port(0), server(NULL), initID(0) {
    callbackHandle = this->registerCallback(CallbackImpl, this);
}

RpcBase::RpcBase(CallbacksManager* callbacksManager) : CallbacksManager(callbacksManager), address(""), port(0), server(NULL), initID(0) {
    callbackHandle = this->registerCallback(CallbackImpl, this);
}

RpcBase::RpcBase() : CallbacksManager(), address(""), port(0), server(NULL), initID(0) {
    callbackHandle = this->registerCallback(CallbackImpl, this);
}

LHandle RpcBase::waitOnConnection() {
    this->initID = 0;
    if (server == NULL) {
        server = new ServerImpl(port, this);
    }
    if (server->connect() == false) {
        return 0;
    }

    this->serverMutex.lock();
    if (this->initID == 0) {
        this->serverCond.wait(this->serverMutex);
    }
    this->serverMutex.unlock();
    return LHandle(this);
}

void RpcBase::init(const char* _address, uint16_t _port) {
    this->initID = 0;
    if (this->address == "") {
        this->address = _address;
        this->port = _port;
    }
}

LHandle RpcBase::connect(const char* address, uint16_t port) {
    this->mutex.lock();
    utils::Client* clientSocket = new Client(address, port);
    LHandle clientSocketID = LHandle(clientSocket);
    if (clientSocket->connect() == false) {
        delete clientSocket;
        clientSocket = NULL;
    }
    if (clientSocket) {
        Writer writer;
        LHandle actionID = LHandle(&writer);

        writer.write(ESTABLISH_CONNECTION);
        writer.write(actionID);
        writer.write(clientSocketID);
        writer.write(this->address.c_str());
        writer.write(this->port);

        Writer writer1;
        writer1.write(RpcBase::MAGIC_NUMBER);
        writer1.write(&writer);
        clientSocket->send(&writer1);
        if (server == NULL) {
            server = new ServerImpl(this->port, this);
        }
        LHandle* remoteClientSocket = (LHandle*)this->waintOnOutput(actionID);
        LHandle remoteClientSocketID = *remoteClientSocket;
        delete remoteClientSocket;
        debug("remoteClientSocketID == %p, clientSocketID == %p \n", remoteClientSocketID.getPtr(), clientSocketID.getPtr());
        this->addresses[clientSocketID] = remoteClientSocketID;

        RemoteConnectionEventImpl rcei(address, port, remoteClientSocketID);
        this->invokeCallbacks(RpcBase::EVENT_REMOTE_CONNECTION, &rcei);
    }
    this->mutex.unlock();
    return LHandle(clientSocket);
}

LHandle RpcBase::establishConnection(const char* address, uint16_t port) {
    if (server != NULL) {
        debug("OK 1\n");
        utils::Client* client = new Client(address, port);
        if (client->connect() == false) {
            delete client;
            return 0;
        }
        this->clinets.push_back(client);
        LHandle clientID = LHandle(client);
        return clientID;
    }
    return 0;
}

LHandle RpcBase::registerCall(const char* name, Function_f function, const utils::ArgumentType* inArgs, uint inArgc,
    const utils::ArgumentType* outArgs, uint outArgc, void* ptr, int rpcParameters) {
    const char* names[] = {name};
    return this->registerCall(names, 1, function, inArgs, inArgc, outArgs, outArgc, ptr, rpcParameters);
}

LHandle RpcBase::registerCall(const char** names, uint namesCount, Function_f function, const utils::ArgumentType* inArgs, uint inArgc,
    const utils::ArgumentType* outArgs, uint outArgc, void* ptr, int rpcParameters) {
    this->callsMutex.lock();
    utils::OglaObject* object = NULL;
    if (namesCount > 1) {
        object = new utils::OglaObject(&names[namesCount - 2], namesCount - 1);
    }
    FunctionProxy* proxy = NULL;
    if (namesCount >= 1) {
        proxy = new FunctionProxy(function, names[namesCount - 1], inArgs, inArgc, object, outArgs, outArgc, ptr);
        Strings strings(names, namesCount);
        debug("function->getName() = %s \n", strings.c_str());
        functionsMap[strings] = std::pair<uint, OglaFunction*>(rpcParameters, proxy);
        registeredFunctions.push_back(proxy);
    }
    this->callsMutex.unlock();
    return LHandle(proxy);
}

void RpcBase::unregisterCall(uint callHandle) {
    this->callsMutex.lock();
    Functions::iterator it = std::find(registeredFunctions.begin(), registeredFunctions.end(), reinterpret_cast<FunctionProxy*> (callHandle));
    if (it != registeredFunctions.end()) {
        FunctionProxy* proxy = reinterpret_cast<FunctionProxy*> (callHandle);
        OglaObject::DestroyObjects(proxy->getRoot());
        delete proxy;
    }
    this->callsMutex.unlock();
}

void RpcBase::call(LHandle connectionID, const char* functionName, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
    const char* names[] = {functionName};
    this->call(connectionID, names, 1, input, args, argc, output);
}

void RpcBase::call(LHandle connectionID, const char** functionNames, uint namesCount, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
    this->call(connectionID, 0, functionNames, namesCount, input, args, argc, output);
}

void RpcBase::call(LHandle connectionID, LHandle functionID, const char** functionNames, uint namesCount,
    utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
    this->mutex.lock();
    if (functionNames != NULL) {
        utils::Client* socket = this->getSocket(connectionID);
        if (socket) {
            LHandle actionID = LHandle(&input);
            Writer writer;
            writer.write(EXECUTE);
            writer.write(actionID);
            writer.write(this->addresses[connectionID]);
            if (functionID == 0) {
                writer.write(0);
                writer.write(namesCount);
                for (uint fa = 0; fa < namesCount; fa++) {
                    writer.write(functionNames[fa]);
                }
            } else {
                writer.write(1);
                writer.write(functionID);
            }
            writer.write(argc);
            for (uint fa = 0; fa < argc; fa++) {
                writer.write(args[fa]);
            }
            writer.write(&(input));
            Writer writer1;
            writer1.write(RpcBase::MAGIC_NUMBER);
            writer1.write(&writer);
            socket->send(&writer1);
            Reader* params = (Reader*) this->waintOnOutput(actionID);
            if (params) {
                output.deserialize(*params);
                delete params;
            }
        } else {
            Strings strings(functionNames, namesCount);
            RpcBase* rpci = (RpcBase*) connectionID.getPtr();
            if (rpci == this && this->functionsMap[strings].second->equals(functionNames[namesCount - 1], args, argc)) {
                this->functionsMap[strings].second->invoke(input, output);
            }
        }
    }
    this->mutex.unlock();
}

void RpcBase::call(LHandle connectionID, LHandle functionHandle, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
    FunctionProxy* proxy = reinterpret_cast<FunctionProxy*> (functionHandle.getPtr());
    const char* functionNam = proxy->getName();
    this->call(connectionID, functionNam, input, args, argc, output);
}

void RpcBase::broadcast(LHandle handle, void* ptr) {
    this->mutex.lock();
    this->outputsMap[handle] = ptr;
    this->cond.broadcast();
    this->mutex.unlock();
}

void* RpcBase::waintOnOutput(LHandle handle) {
    this->mutex.lock();
    if (outputsMap.count(handle) == 0) {
        this->cond.wait(&(this->mutex));
    }
    void* ptr = outputsMap[handle];
    outputsMap.erase(outputsMap.find(handle));
    this->mutex.unlock();
    return ptr;
}

void RpcBase::InvokeFunction(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc) {
    Writer paramsOut;
    rpc->executeFunction(reader, paramsOut, actionID, connectionID);
    rpc->inputsDict.remove(actionID);

    Writer writer;
    writer.write(RETURN);
    writer.write(actionID);
    writer.write(rpc->addresses[connectionID]);
    writer.write(&paramsOut);

    Writer writer1;
    writer1.write(RpcBase::MAGIC_NUMBER);
    writer1.write(&writer);

    utils::Client* clientSocket = rpc->getSocket(connectionID);
    if (clientSocket) {
        clientSocket->send(&writer1);
    }
}

void RpcBase::GetOutput(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc) {
    rpc->inputsDict.add(actionID, &reader);
}

void RpcBase::Return(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc) {
    Reader* params = new Reader();
    reader.read(params);
    if (params->getSize() > 0) {
        rpc->broadcast(actionID, (void*) params);
    } else {
        rpc->broadcast(actionID, NULL);
        delete params;
    }
}

void RpcBase::EstablishConnection(LHandle actionID, LHandle remoteClientID, Reader& reader, RpcBase* rpc) {
    Writer writer;
    char* address = reader.readStr();
    int port = reader.readInt();
    LHandle clientID = rpc->establishConnection(address, port);
    rpc->addresses[clientID] = (remoteClientID);
    writer.write(SET_CLIENT_ID);
    writer.write(actionID);
    writer.write(clientID);
    debug("remoteClientSocketID == %p, clientSocketID == %p \n", remoteClientID.getPtr(), clientID.getPtr());
    Client* clientSocket = rpc->getSocket(clientID);
    Writer writer1;
    writer1.write(RpcBase::MAGIC_NUMBER);
    writer1.write(&writer);
    if (clientSocket) {
        clientSocket->send(&writer1);
    }
    RemoteConnectionEventImpl rce(address, port, clientID);
    rpc->invokeCallbacks(RpcBase::RpcBase::EVENT_REMOTE_CONNECTION, &rce);
    delete[] address;
}

void RpcBase::SetMyClientID(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc) {
    debug("MyClientID == %p \n", connectionID.getPtr());
    rpc->broadcast(actionID, (void*) new LHandle(connectionID));
}

void RpcBase::GetRegisteredFunctions(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc) {
    rpc->sendFunctionEvent(actionID, connectionID);
}
}