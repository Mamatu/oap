/* 
 * File:   RPCImplementation.cpp
 * Author: marcin
 * 
 * Created on 15 January 2013, 22:14
 */

#include "RpcImpl.h"
#include "Writer.h"
#include "Reader.h"
#include "Buffer.h"
#include <algorithm>

namespace utils {

#define CALL(func,...) RpcBase::func(__VA_ARGS__);
#define CALL_RET(func,...) return RpcBase::func(__VA_ARGS__);

    RpcImpl::RpcImpl() : RpcBase() {
    }

    RpcImpl::RpcImpl(CallbacksManager* callbacksManager) : RpcBase(callbacksManager) {
    }

    RpcImpl::RpcImpl(Callbacks* callbacks) : RpcBase(callbacks) {
    }

    RpcImpl::RpcImpl(const char* _address, int16_t _port) : RpcBase(_address, _port) {
    }

    RpcImpl::RpcImpl(const char* _address, int16_t _port, CallbacksManager* callbacksManager) : RpcBase(_address, _port, callbacksManager) {
    }

    RpcImpl::RpcImpl(const char* _address, int16_t _port, Callbacks* callbacks) : RpcBase(_address, _port, callbacks) {
    }

    RpcImpl::~RpcImpl() {
    }

    LHandle RpcImpl::waitOnConnection() {
        CALL_RET(waitOnConnection);
    }

    int RpcImpl::init(const char* address, int16_t port) {
        CALL(init, address, port);
    }

    LHandle RpcImpl::connect(const char* address, int16_t port) {
        CALL_RET(connect, address, port);
    }

    void RpcImpl::call(LHandle connectionID, const char** names, int naemsCount, utils::Writer& input, utils::ArgumentType* args, int argc, utils::Reader& output) {
        CALL(call, connectionID, names, naemsCount, input, args, argc, output);
    }

    void RpcImpl::call(LHandle connectionID, const char* functionName, utils::Writer& input, utils::ArgumentType* args, int argc, utils::Reader& output) {
        CALL(call, connectionID, functionName, input, args, argc, output);
    }

    void RpcImpl::call(LHandle connectionID, LHandle functionID, utils::Writer& input, utils::ArgumentType* args, int argc, utils::Reader& output) {
        CALL(call, connectionID, functionID, input, args, argc, output);
    }

    void Function_impl(utils::Reader& input, utils::Writer* outputs, void * ptr) {
        utils::OapFunction* function = (utils::OapFunction*)(ptr);
        if (outputs) {
            function->invoke(input, *outputs);
        } else {
            function->invoke(input);
        }
    }

    LHandle RpcImpl::registerCall(const char** names, int namesCount, Function_f function, const utils::ArgumentType* inArgs, int inArgc,
            const utils::ArgumentType* outArgs, int outArgc, void* ptr, int rpcParameters) {
        CALL_RET(registerCall, names, namesCount, function, inArgs, inArgc, outArgs, outArgc, ptr, rpcParameters)
    }

    LHandle RpcImpl::registerCall(const char* functionName, Function_f function, const utils::ArgumentType* inArgs, int inArgc,
            const utils::ArgumentType* outArgs, int outArgc, void* ptr, int rpcParameters) {
        CALL_RET(registerCall, functionName, function, inArgs, inArgc, outArgs, outArgc, ptr, rpcParameters)
    }

    LHandle RpcImpl::registerCall(utils::OapFunction* function) {
        const char* functionName = function->getName();
        const char** names = NULL;
        int namesCount = 0;
        OapObject::ConvertToStrings(function->getRoot(), &names, namesCount);
        ArrayTools::add(&names, namesCount, functionName);
        LHandle id = this->registerCall(names, namesCount, Function_impl, function->getInputArgumentsTypes(), function->getInputArgc(),
                function->getOutputArgumentsTypes(), function->getOutputArgc(), function);
        delete[] names;
        return id;
    }

    void RpcImpl::unregisterCall(int callHandle) {
        CALL(unregisterCall, callHandle)
    }
};


