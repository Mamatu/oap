/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "ThreadUtils.h"

namespace oap {
namespace utils {

Thread::Thread() : m_iscond(false), m_isonethread(true) {}

Thread::~Thread() {}

void Thread::onRun(pthread_t threadId) {}

void Thread::setFunction(ThreadFunction_f _function, void* _ptr) {
  this->m_function = _function;
  this->m_ptr = _ptr;
}

void* Thread::Execute(void* ptr) {
  Thread* thread = (Thread*)(ptr);
  if (!thread->m_isonethread) {
    thread->m_mutex.lock();
    if (thread->m_iscond == false) {
      thread->m_cond.wait(thread->m_mutex);
      thread->m_iscond = false;
    }
    thread->m_mutex.unlock();
  }
  thread->m_function(thread->m_ptr);

  if (!thread->m_isonethread) {
    void* retVal = NULL;
    pthread_exit(retVal);
  }
}

void Thread::run(bool inTheSameThreead) {
  m_isonethread = inTheSameThreead;
  if (!m_isonethread) {
    pthread_create(&m_thread, 0, Thread::Execute, this);
  } else {
    m_thread = pthread_self();
  }
  onRun(m_thread);
  if (!m_isonethread) {
    m_mutex.lock();
    m_iscond = true;
    m_cond.signal();
    m_mutex.unlock();
  } else {
    Execute(this);
  }
}

void Thread::join() {
  if (m_isonethread == false) {
    void* o = NULL;
    pthread_join(m_thread, &o);
  }
}

namespace sync {

Mutex::Mutex() : ptr_mutex(NULL) {
  pthread_mutex_init(&mutex, 0);
  ptr_mutex = &mutex;
}

Mutex::~Mutex() {
  if (ptr_mutex != NULL) {
    pthread_mutex_destroy(ptr_mutex);
  }
}

void Mutex::lock() { pthread_mutex_lock(ptr_mutex); }

void Mutex::unlock() { pthread_mutex_unlock(ptr_mutex); }

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

Cond::Cond() { pthread_cond_init(&cond, 0); }

Cond::~Cond() { pthread_cond_destroy(&cond); }

void Cond::wait(Mutex* mutex) {
  if (mutex != NULL) {
    pthread_cond_wait(&cond, mutex->ptr_mutex);
  }
}

void Cond::wait(Mutex& mutex) { pthread_cond_wait(&cond, mutex.ptr_mutex); }

void Cond::broadcast() { pthread_cond_broadcast(&cond); }

void Cond::signal() { pthread_cond_signal(&cond); }

Barrier::Barrier() : m_init(false) {}

Barrier::~Barrier() {
  if (m_init) {
    pthread_barrier_destroy(&m_barrier);
    m_init = false;
  }
}

Barrier::Barrier(unsigned int count) : m_init(false) { this->init(count); }

void Barrier::init(unsigned int count) {
  if (m_init) {
    pthread_barrier_destroy(&m_barrier);
  }
  pthread_barrier_init(&m_barrier, NULL, count);
  m_init = true;
}

void Barrier::wait() { pthread_barrier_wait(&m_barrier); }

Semaphore::Semaphore() { sem_init(&m_sem, 0, 0); }
Semaphore::~Semaphore() { sem_destroy(&m_sem); }
void Semaphore::wait() { sem_wait(&m_sem); }
void Semaphore::signal() { sem_post(&m_sem); }

CondBool::CondBool() { m_shouldlocked = true; }
CondBool::~CondBool() {}

void CondBool::wait() {
  m_mutex.lock();
  if (m_shouldlocked) {
    m_cond.wait(m_mutex);
  }
  m_shouldlocked = true;
  m_mutex.unlock();
}

void CondBool::signal() {
  m_mutex.lock();
  m_shouldlocked = false;
  m_cond.signal();
  m_mutex.unlock();
}
void CondBool::broadcast() {
  m_mutex.lock();
  m_shouldlocked = false;
  m_cond.broadcast();
  m_mutex.unlock();
}
}
}
}
