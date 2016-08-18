
#ifndef RPCIMPLEMENTATION_H
#define	RPCIMPLEMENTATION_H

#include "RpcBase.h"

namespace utils {

    class RpcImpl : public RpcBase {
    public:
        RpcImpl();
        RpcImpl(CallbacksManager* callbacksManager);
        RpcImpl(Callbacks* callbacks);
        RpcImpl(const char* _address, int16_t _port);
        RpcImpl(const char* _address, int16_t _port, CallbacksManager* callbacksManager);
        RpcImpl(const char* _address, int16_t _port, Callbacks* callbacks);
        
        virtual ~RpcImpl();

        LHandle waitOnConnection();
        int init(const char* address, int16_t port);

        LHandle connect(const char* address, int16_t port);

        void call(LHandle connectionID, const char** names, int naemsCount, utils::Writer& input, utils::ArgumentType* args, int argc, utils::Reader& output);

        void call(LHandle connectionID, const char* functionName, utils::Writer& input, utils::ArgumentType* args, int argc, utils::Reader& output);

        void call(LHandle connectionID, LHandle functionID, utils::Writer& input, utils::ArgumentType* args, int argc, utils::Reader& output);
        
        LHandle registerCall(const char** names, int namesCount, Function_f function, const utils::ArgumentType* inArgs, int inArgc,
                const utils::ArgumentType* outArgs = NULL, int outArgc = 0, void* ptr = NULL, int rpcParameters = 0);

        LHandle registerCall(const char* functionName, Function_f function, const utils::ArgumentType* inArgs, int inArgc,
                const utils::ArgumentType* outArgs = NULL, int outArgc = 0, void* ptr = NULL, int rpcParameters = 0);

        LHandle registerCall(utils::OapFunction* function);

        void unregisterCall(int callHandle);
    };
}
#endif	/* RPCIMPLEMENTATION_H */

