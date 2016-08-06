/* 
 * File:   APIReader.h
 * Author: marcin
 *
 * Created on 03 January 2013, 21:22
 */

#ifndef OGLASERVER_H
#define	OGLASERVER_H

#include <map>
#include "DebugLogs.h"
#include "RpcImpl.h"
#include "Argument.h"
#include "Status.h"
#include "OapInterface.h"

namespace core {
class Pair;
class ObjectLoader;
class ExtensionLoader;
typedef std::pair<std::string, std::vector<std::string> > ModuleInfo;
typedef std::vector<ModuleInfo> ModulesInfos;

class OapServer : public OapInterface {
private:
    Status registerFunction(utils::OapFunction* function,
        std::vector<std::string>* names, void* ptr, uint& functionHandle);
    Status registerObject(utils::OapObject* object,
        std::vector<std::string>* names, void* ptr, uint& objectHandle);
    void registerFunctions(utils::Container<utils::OapFunction>* functionsContainer);
    void registerObjects(utils::Container<utils::OapObject>* objectsContainer);
    static void Execute(void* ptr);
    bool destroyed;
    utils::Thread mainThread;
public:
    OapServer(int port, bool registerDefaultMethods = true);
    virtual ~OapServer();
    Status registerFunction(utils::OapFunction* function);

    Status executeFunction(LHandle connectionID, const char* functionName,
        utils::Writer& input,
        utils::ArgumentType* args,
        uint argc,
        utils::Reader& output);

    Status executeFunction(LHandle connectionID,
        const char** names, uint namesCount,
        utils::Writer& input,
        utils::ArgumentType* args,
        uint argc,
        utils::Reader& output);

    Status executeFunction(LHandle connectionID,
        LHandle functionHandle,
        utils::Writer& input,
        utils::ArgumentType* args, uint argc,
        utils::Reader& output);

    static int EVENT_REGISTER_FUNCTION;
    static int EVENT_UNREGISTER_FUNCTION;
    static int EVENT_REMOTE_CONNECTION;
    static int EVENT_REGISTER_OBJECT;
    static int EVENT_UNREGISTER_OBJECT;

    int registerModules(const char* path, LHandle& handle);

    int unregisterModule(LHandle handle);
    void wait();
private:
    typedef std::vector<utils::OapFunction*> Functions;
    Functions functions;

    friend class ObjectLoader;
    friend class ExtensionLoader;
    friend class CallbackImpl1;
    friend class CallbackImpl2;

    static void ModuleCallback(int event, void* object, void* userPtr);
    static void RegisterObject(utils::Reader& input, utils::Writer* output,
        void* ptr);
    static void UnregisterObject(utils::Reader& input, utils::Writer* output,
        void* ptr);

    ObjectLoader* moduleLoader;
    ExtensionLoader* extensionLoader;
    utils::RpcImpl* rpc;

    uint remoteRpcHandle;
    void destroy();

    typedef std::vector<uint> HandlesList;
    HandlesList miHandles;
};
}

#endif	/* APIReader_H */

