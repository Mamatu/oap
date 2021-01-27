/*
 * Copyright 2016 - 2021 Marcin Matula
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
namespace thread
{
std::string str_id (const std::thread* thread)
{
  std::stringstream sstr;
  sstr << thread->get_id();
  return sstr.str();
}

std::string str_id (std::thread::id id)
{
  std::stringstream sstr;
  sstr << id;
  return sstr.str();
}

std::string str_id ()
{
  std::stringstream sstr;
  sstr << std::this_thread::get_id();
  return sstr.str();
}
}
namespace utils {

AsyncQueue::AsyncQueue ()
{
  m_stop.store (false);
}

AsyncQueue::~AsyncQueue ()
{
  stop();
}

bool AsyncQueue::push (const Function& function)
{
  return _push (function);
}

bool AsyncQueue::push (Function&& function)
{
  return _push (function);
}

void AsyncQueue::stop ()
{
  if (!m_stop)
  {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_stop = true;
    }
    m_cv.notify_one();
    if (m_thread != nullptr)
    {
      m_thread->join();
    }
    delete m_thread;
  }
}

const std::thread* AsyncQueue::getThread() const
{
  return m_thread;
}

void AsyncQueue::runThread ()
{
  if (m_thread == nullptr)
  {
    m_thread = new std::thread ([this]()
      {
        bool cont = true;
        do
        {
          Function function;
          {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_queue.empty())
            {
              function = std::move (m_queue.front());
              m_queue.pop();
            }
          }
          function (std::this_thread::get_id());
          {
            std::unique_lock<std::mutex> ul(m_mutex);
            m_cv.wait (ul, [this]() { return m_stop.load() || m_queue.size() > 0; });
          }
          {
            std::lock_guard<std::mutex> lock(m_mutex);
            cont = !m_queue.empty() || !m_stop.load();
          }
        } while (cont);
      });
  }
}

Thread::Thread() {}

Thread::~Thread()
{
  stop();
}

void Thread::onRun(std::thread::id threadId) {}

void Thread::run (Function _function, void* _ptr)
{
  m_asyncQueue.push ([this, _function, _ptr](std::thread::id id)
      {
        {
          std::unique_lock<std::mutex> ul(m_mutex);
          m_cv.wait (ul, [this]() { return m_onrunDone; });
        }
        _function (_ptr);
      });
  onRun (m_asyncQueue.getThread()->get_id());
  {
    std::unique_lock<std::mutex> ul(m_mutex);
    m_onrunDone = true;
  }
  m_cv.notify_one();
}

void Thread::stop()
{
  m_asyncQueue.stop();
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
  oap::utils::sync::MutexLocker lock (m_mutex);
  if (m_shouldlocked) {
    m_cond.wait(m_mutex);
  }
  m_shouldlocked = true;
}

void CondBool::signal() {
  oap::utils::sync::MutexLocker lock (m_mutex);
  m_shouldlocked = false;
  m_cond.signal();
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

namespace std
{
  std::string to_string(const std::thread* thread)
  {
    return oap::thread::str_id (thread);
  }

  std::string to_string(std::thread::id id)
  {
    return oap::thread::str_id (id);
  }
}
