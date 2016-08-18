
#include "Async.h"
#include <pthread.h>

namespace utils {

    int AsyncQueue::EVENT_ACTION_DONE = 1;

    AsyncQueue::AsyncQueue(Callbacks* callbacks) : CallbacksManager(callbacks), stoped(false) {
        pthread_create(&thread, 0, AsyncQueue::ExecuteMainThread, this);
    }

    AsyncQueue::~AsyncQueue() {
        this->stop();
    }

    void AsyncQueue::stop() {
        this->mutex.lock();
        this->stoped = true;
        this->cond.signal();
        this->mutex.unlock();
        void* o;
        pthread_join(thread, &o);
    }

    void* AsyncQueue::ExecuteMainThread(void* ptr) {
        AsyncQueue* async = (AsyncQueue*) ptr;
        async->mutex.lock();
        while (async->stoped == false) {
            AsyncQueue::Action* action = async->actions.front();
            async->actions.pop();
            async->mutex.unlock();
            action->action(action->userPtr);
            void* actionID = reinterpret_cast<void*> (action);
            async->callbacks.invoke(AsyncQueue::EVENT_ACTION_DONE, &actionID);
            async->mutex.lock();
            if (async->actions.size() == 0 && async->stoped == false) {
                async->cond.wait(async->mutex);
            }
        }
        async->mutex.unlock();
        pthread_exit(0);
    }

    LHandle AsyncQueue::execute(Action_f action, void* userPtr) {
        AsyncQueue::Action* actionPtr = new AsyncQueue::Action();
        actionPtr->action = action;
        actionPtr->userPtr = userPtr;
        this->mutex.lock();
        this->actions.push(actionPtr);
        this->mutex.unlock();
        LHandle id = LHandle(actionPtr);
        return id;
    }

    AsyncProcess::AsyncProcess(AsyncQueue* async) : CallbacksManager(false), sharedAsync(false) {
        if (async) {
            this->async = async;
            this->sharedAsync = true;
        } else {
            this->async = new AsyncQueue();
        }
        this->setCallbacksManager(this->async);
    }

    AsyncProcess::~AsyncProcess() {
        if (this->sharedAsync == false) {
            delete this->async;
        }
        this->async = NULL;
    }

    AsyncQueue& AsyncProcess::getInstance() const {
        return *async;
    }

}
