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
        utils::sync::Mutex mutex;

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
