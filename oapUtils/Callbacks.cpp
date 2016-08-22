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




#include "Callbacks.h"

namespace utils {

    Callbacks::Callbacks() {
    }

    Callbacks::Callbacks(const Callbacks& orig) {
    }

    Callbacks::~Callbacks() {
        mutex.lock();
        for (uint fa = 0; fa < callbackInfos.size(); fa++) {
            delete callbackInfos[fa];
        }
        mutex.unlock();
    }

    LHandle Callbacks::add(Callback_f callback, void* userPtr) {
        mutex.lock();
        CallbackInfo* callbackInfo = new CallbackInfo();
        callbackInfo->callback = callback;
        callbackInfo->userPtr = userPtr;
        this->callbackInfos.push_back(callbackInfo);
        LHandle id = LHandle(callbackInfo);
        mutex.unlock();
        return id;
    }

    void Callbacks::remove(LHandle callbackID) {
        CallbackInfo* callbackInfo = reinterpret_cast<CallbackInfo*> (callbackID.getPtr());
        mutex.lock();
        CallbackInfos::iterator it = std::find(this->callbackInfos.begin(), this->callbackInfos.end(), callbackInfo);
        if (it != this->callbackInfos.end()) {
            this->callbackInfos.erase(it);
            delete callbackInfo;
        }
        mutex.unlock();
    }

    void Callbacks::invoke(int event, void* object) {
        mutex.lock();
        for (uint fa = 0; fa<this->callbackInfos.size(); fa++) {
            CallbackInfo* ptr = this->callbackInfos[fa];
            ptr->callback(event, object, ptr->userPtr);
        }
        mutex.unlock();
    }

    CallbacksManager::CallbacksManager() : callbacks(new Callbacks()), sharedCallbacks(false) {
    }

    CallbacksManager::CallbacksManager(bool createCallbacks) : callbacks(NULL), sharedCallbacks(false) {
        if (createCallbacks) {
            this->callbacks = new Callbacks();
        }
    }

    CallbacksManager::CallbacksManager(Callbacks* callbacks) : sharedCallbacks(false) {
        if (callbacks) {
            this->callbacks = callbacks;
            this->sharedCallbacks = true;
        }
    }

    CallbacksManager::CallbacksManager(CallbacksManager* callbacksManager) : sharedCallbacks(false) {
        if (callbacksManager) {
            this->callbacks = callbacksManager->callbacks;
            this->sharedCallbacks = true;
        } else {
            this->callbacks = new Callbacks();
        }
    }

    CallbacksManager::~CallbacksManager() {
        if (sharedCallbacks == false && callbacks) {
            delete callbacks;
        }
    }

    LHandle CallbacksManager::registerCallback(Callback_f callback, void* userPtr) {
        if (this->callbacks == NULL) {
            return 0;
        }
        return this->callbacks->add(callback, userPtr);
    }

    int CallbacksManager::unregisterCallback(LHandle callbackID) {
        if (this->callbacks == NULL) {
            return 1;
        }
        this->callbacks->remove(callbackID);
        return 0;
    }

    void CallbacksManager::setCallbacks(Callbacks* callbacks) {
        this->callbacks = callbacks;
        this->sharedCallbacks = true;
    }

    void CallbacksManager::setCallbacksManager(CallbacksManager* callbacksManager) {
        this->setCallbacks(callbacksManager->callbacks);
    }

    void CallbacksManager::invokeCallbacks(int event, void* object) {
        if (this->callbacks == NULL) {
            return;
        }
        this->callbacks->invoke(event, object);
    }

}
