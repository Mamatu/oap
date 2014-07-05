/* 
 * File:   mutex.cpp
 * Author: marcin
 * 
 * Created on 26 December 2012, 20:38
 */

#include "ThreadUtils.h"

namespace utils {

    Thread::Thread() {
    }

    Thread::~Thread() {
    }

    void Thread::setFunction(ThreadFunction_f _function, void* _ptr) {
        this->function = _function;
        this->ptr = _ptr;
    }

    void* Thread::Execute(void* ptr) {
        Thread* thread = (Thread*) (ptr);
        thread->function(thread->ptr);
        void* retVal = NULL;
        if (thread->externalThread) {
            pthread_exit(retVal);
        }
    }

    void Thread::run(bool inTheSameThread) {
        externalThread = !inTheSameThread;
        if (externalThread == true) {
            pthread_create(&thread, 0, Thread::Execute, this);
        } else {
            Thread::Execute(this);
        }
    }

    void Thread::yield() {
        if (externalThread == true) {
            void* o = NULL;
            pthread_join(thread, &o);
        }
    }
}

namespace synchronization {

    Mutex::Mutex() : ptr_mutex(NULL) {
        pthread_mutex_init(&mutex, 0);
        ptr_mutex = &mutex;
    }

    Mutex::~Mutex() {
        if (ptr_mutex != NULL) {
            pthread_mutex_destroy(ptr_mutex);
        }
    }

    void Mutex::lock() {
        pthread_mutex_lock(ptr_mutex);
    }

    void Mutex::unlock() {
        pthread_mutex_unlock(ptr_mutex);
    }

    RecursiveMutex::RecursiveMutex() : Mutex() {
        pthread_mutexattr_init(&mutexattr);
        pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&mutex, &mutexattr);
    }

    RecursiveMutex::~RecursiveMutex() {
        if (ptr_mutexattr != NULL) {
            pthread_mutexattr_destroy(ptr_mutexattr);
        }
    }

    Cond::Cond() {
        pthread_cond_init(&cond, 0);
    }

    Cond::~Cond() {
        pthread_cond_destroy(&cond);
    }

    void Cond::wait(Mutex* mutex) {
        if (mutex != NULL) {
            pthread_cond_wait(&cond, mutex->ptr_mutex);
        }
    }

    void Cond::wait(Mutex& mutex) {
        pthread_cond_wait(&cond, mutex.ptr_mutex);
    }

    void Cond::broadcast() {
        pthread_cond_broadcast(&cond);
    }

    void Cond::signal() {
        pthread_cond_signal(&cond);
    }

    Barrier::Barrier() : ptr_barrier(NULL) {
    }

    Barrier::~Barrier() {
        if (ptr_barrier) {
            pthread_barrier_destroy(ptr_barrier);
            ptr_barrier = NULL;
        }
    }

    Barrier::Barrier(int count) : ptr_barrier(NULL) {
        this->init(count);
    }

    void Barrier::init(int count) {
        if (this->ptr_barrier != NULL) {
            pthread_barrier_destroy(ptr_barrier);
            ptr_barrier = NULL;
        }
        pthread_barrier_init(&(this->barrier), 0, count);
        this->ptr_barrier = &(this->barrier);
    }

    void Barrier::wait() {
        pthread_barrier_wait(this->ptr_barrier);
    }
}