/* 
 * File:   mutex.h
 * Author: marcin
 *
 * Created on 26 December 2012, 20:38
 */

#ifndef OGLA_THREAD_UTILS_H
#define	OGLA_THREAD_UTILS_H

#include <pthread.h>
#include "DebugLogs.h"

namespace utils {

    typedef void (*ThreadFunction_f)(void* ptr);

    class Thread {
        bool externalThread;
        ThreadFunction_f function;
        void* ptr;
        pthread_t thread;
        static void* Execute(void* ptr);
    public:
        Thread();
        virtual ~Thread();
        void setFunction(ThreadFunction_f _function, void* _ptr);
        void run(bool inTheSameThread = false);
        void yield();
    };
}

namespace synchronization {
    class Cond;

    class Mutex {
    public:
        Mutex();
        virtual ~Mutex();
        void lock();
        void unlock();
    protected:
        pthread_mutex_t mutex;
        pthread_mutex_t* ptr_mutex;
        friend class Cond;
    };

    class RecursiveMutex : public Mutex {
    public:
        RecursiveMutex();
        virtual ~RecursiveMutex();
    private:
        pthread_mutexattr_t mutexattr;
        pthread_mutexattr_t* ptr_mutexattr;
    };

    class Cond {
    public:
        Cond();
        virtual ~Cond();
        void wait(Mutex* mutex);
        void wait(Mutex& mutex);
        void broadcast();
        void signal();
    private:
        pthread_cond_t cond;
    };

    class Barrier {
        pthread_barrier_t barrier;
        pthread_barrier_t* ptr_barrier;
    public:
        Barrier();
        Barrier(int count);
        void init(int count);
        ~Barrier();
        void wait();
    };
}
#endif

