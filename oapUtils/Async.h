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




#ifndef ASYNC_H
#define	ASYNC_H

#include "Callbacks.h"
#include "ThreadUtils.h"
#include <queue>

namespace utils {
    typedef void (*Action_f)(void* userPtr);

    class AsyncQueue : public CallbacksManager {
    public:
        static int EVENT_ACTION_DONE;

        AsyncQueue(Callbacks* callbacks = NULL);
        virtual ~AsyncQueue();

        LHandle execute(Action_f action, void* userPtr);

    private:
        void stop();

        class Action {
        public:
            Action_f action;
            void* userPtr;
        };

        pthread_t thread;
        utils::sync::Mutex mutex;
        utils::sync::Cond cond;
        bool stoped;
        std::queue<Action*> actions;

        static void* ExecuteMainThread(void* ptr);
        Callbacks callbacks;
    };

    class AsyncProcess : public CallbacksManager {
    public:
        AsyncProcess(AsyncQueue* async = NULL);
        virtual ~AsyncProcess();
    protected:
        AsyncQueue& getInstance() const;
    private:
        bool sharedAsync;
        AsyncQueue* async;
    };

}

#endif	/* ASYNCRANDOMS_H */
