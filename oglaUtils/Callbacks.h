/* 
 * File:   Callbacks.h
 * Author: mmatula
 *
 * Created on August 15, 2013, 9:20 PM
 */

#ifndef CALLBACKS_H
#define	CALLBACKS_H

#include <vector>
#include <algorithm>
#include "ThreadUtils.h"
#include "LHandle.h"

namespace utils {

    typedef void (Callback_f) (int event, void* object, void* userPtr);

    class Callbacks {
    public:
        Callbacks();
        virtual ~Callbacks();

        LHandle add(Callback_f callback, void* userPtr);
        void remove(LHandle callbackID);
        void invoke(int event, void* object);

    private:
        synchronization::Mutex mutex;

        struct CallbackInfo {
            Callback_f* callback;
            void* userPtr;
        };

        typedef std::vector<CallbackInfo*> CallbackInfos;
        CallbackInfos callbackInfos;

        Callbacks(const Callbacks& orig);
    };

    class CallbacksManager {
    public:
        CallbacksManager();
        CallbacksManager(bool createCallbacks);
        CallbacksManager(Callbacks* callbacks);
        CallbacksManager(CallbacksManager* callbacksManager);
        virtual ~CallbacksManager();

        LHandle registerCallback(Callback_f callback, void* userPtr);
        int unregisterCallback(LHandle callbackID);

    protected:
        void invokeCallbacks(int event, void* object);
        void setCallbacks(Callbacks* callbacks);
        void setCallbacksManager(CallbacksManager* callbacksManager);
    private:
        Callbacks* callbacks;
        bool sharedCallbacks;
    };
}
#endif	/* CALLBACKS_H */

