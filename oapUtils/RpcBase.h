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




#ifndef RPCINTERFACE_H
#define	RPCINTERFACE_H

#include <vector>
#include <map>

#include "Socket.h"
#include "Reader.h"
#include "Writer.h"
#include "Argument.h"
#include "WrapperInterfaces.h"
#include "ThreadUtils.h"


#include <vector>
#include <map>

#include "Socket.h"
#include "Reader.h"
#include "Writer.h"
#include "Callbacks.h"
#include "Argument.h"
#include "WrapperInterfaces.h"
#include "ThreadUtils.h"

namespace utils {

    class Strings;
    class InputsDict;

    /**
     *  0 - 4 magic number \ begin of section
     *  4 - 8 size of ONLY data section, this 4 bytes are NOT contained in data section
     *  8 - 12 : type of action,
     *  12 - 16 : id of action
     *  16 - 20 : id of client
     *  20 - x: data section
     * 
     */
    class RpcBase : public CallbacksManager {
        static const int MAGIC_NUMBER;
    public:
        static const int EVENT_REGISTER_FUNCTION;
        static const int EVENT_UNREGISTER_FUNCTION;
        static const int EVENT_REMOTE_CONNECTION;

        class RemoteConnectionEvent {
        protected:
            RemoteConnectionEvent();
            virtual ~RemoteConnectionEvent();
        public:
            virtual void getAddress(char*& address) const = 0;
            virtual uint16_t getPort() const = 0;
            virtual LHandle getConnectionID() const = 0;
        };

        RpcBase();
        RpcBase(CallbacksManager* callbacksManager);
        RpcBase(Callbacks* callbacks);
        RpcBase(const char* _address, uint16_t _port);
        RpcBase(const char* _address, uint16_t _port, CallbacksManager* callbacksManager);
        RpcBase(const char* _address, uint16_t _port, Callbacks* callbacks);
        virtual ~RpcBase();

    protected:
        LHandle waitOnConnection();
        void init(const char* address, uint16_t port);

        LHandle connect(const char* address, uint16_t port);

        void call(LHandle connectionID, 
            const char** names,  uint naemsCount, 
            utils::Writer& input, utils::ArgumentType* args, 
            uint argc, utils::Reader& output);

        void call(LHandle connectionID, const char* functionName,
            utils::Writer& input, utils::ArgumentType* args, uint argc, 
            utils::Reader& output);

        void call(LHandle connectionID, 
            LHandle functionID, 
            utils::Writer& input, utils::
            ArgumentType* args, uint argc, utils::Reader& output);

        void getRegisteredFunctions(uint connectionID);

        LHandle registerCall(const char** names, uint namesCount, 
            Function_f function, const utils::ArgumentType* inArgs, uint inArgc,
            const utils::ArgumentType* outArgs = NULL, uint outArgc = 0, 
            void* ptr = NULL, int rpcParameters = 0);

        LHandle registerCall(const char* functionName, Function_f function, 
            const utils::ArgumentType* inArgs, uint inArgc,
            const utils::ArgumentType* outArgs = NULL, 
            uint outArgc = 0, void* ptr = NULL, int rpcParameters = 0);

        void unregisterCall(uint callHandle);
        
    private:
        void call(LHandle connectionID, LHandle functionID, 
            const char** names, uint naemsCount, 
            utils::Writer& input, utils::ArgumentType* args, 
            uint argc, utils::Reader& output);
        
        LHandle callbackHandle;

        class Strings {
            std::string all;
        public:
            std::vector<std::string> stringsVec;
            Strings(const char* str);
            Strings(const char** strs, uint count);
            Strings(const std::vector<std::string>& stringsVec);

            const char* c_str();
            bool operator<(const Strings& strings) const;
        };

        class InputsDict {
            typedef std::map<LHandle, Writer*> InputsMap;
            InputsMap inputsMap;
            utils::sync::RecursiveMutex mutex;
        public:
            void add(LHandle id, Serializable* serialize);
            int count(LHandle id);
            Reader* get(LHandle id);
            void remove(LHandle id);
        };

        std::string address;
        uint16_t port;
        uint initID;

        utils::Server* server;
        class ServerImpl;
        friend class ServerImpl;
        std::vector<utils::Socket*> clinets;

        utils::sync::RecursiveMutex mutex;
        utils::sync::RecursiveMutex callsMutex;
        utils::sync::Cond cond;
        utils::sync::Cond serverCond;
        utils::sync::Mutex serverMutex;

        typedef std::map<Strings, std::pair<int, OapFunction*> > FunctionsMap;

        struct SocketData {
            Writer* writer;
            char* buffer;
        };
        typedef std::map<Socket*, SocketData> SocketsWriters;
        typedef std::vector<OapFunction*> Functions;
        typedef std::map<LHandle, void*> OutpusMap;
        typedef std::map<LHandle, LHandle> Addresses;

        friend class Strings;
        FunctionsMap functionsMap;
        Functions registeredFunctions;
        OutpusMap outputsMap;
        SocketsWriters socketsWriters;
        InputsDict inputsDict;
        Addresses addresses;
        utils::sync::Mutex outputsMutex;

        void broadcast(LHandle id, void* ptr);
        void* waintOnOutput(LHandle id);

        static void SerializeFunctionInfo(const char** names, uint namesCount, utils::ArgumentType* inArgs, uint inArgc,
                utils::ArgumentType* outArgs, uint outArgc, utils::Writer * writer);

        void executeFunction(Reader& reader, utils::Writer& writer, LHandle actionID, LHandle connectionID);
        LHandle establishConnection(const char* address, uint16_t port);

        static void DeserializeArgumentsTypes(ArgumentType** types, uint& arguemntsNumber, Reader& reader);
        static void InvokeFunction(LHandle actionID, LHandle clientID, Reader& reader, RpcBase* rpc);
        static void GetOutput(LHandle actionID, LHandle clientID, Reader& reader, RpcBase* rpc);
        static void Return(LHandle actionID, LHandle clientID, Reader& reader, RpcBase* rpc);
        static void EstablishConnection(LHandle actionID, LHandle clientID, Reader& reader, RpcBase* rpc);
        static void SetMyClientID(LHandle actionID, LHandle clientID, Reader& reader, RpcBase* rpc);
        static void RegisterFunctionEvent(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc);
        static void UnregisterFunctionEvent(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc);
        static void GetRegisteredFunctions(LHandle actionID, LHandle connectionID, Reader& reader, RpcBase* rpc);

        static void ServerSubExecution(RpcBase* rpc, int type, Reader& reader);

        void sendFunctionEvent(LHandle socketID, LHandle clientID);
        void putFunctionEvent(Writer& writer, LHandle socketID, LHandle clientID);

        static void CallbackImpl(int event, void* eventObj, void* user_ptr);
        utils::Client* getSocket(LHandle clinetID);
        RpcBase(const RpcBase& orig);
    };
}
#endif	/* RPCINTERFACE_H */