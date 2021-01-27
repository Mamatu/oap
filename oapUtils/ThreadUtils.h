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

#ifndef OAP_THREAD_UTILS_H
#define OAP_THREAD_UTILS_H

#include <pthread.h>
#include <semaphore.h>

#include <atomic>
#include <functional>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "Logger.h"

namespace oap {

namespace thread
{
std::string str_id (const std::thread* thread);

std::string str_id (std::thread::id id);

std::string str_id ();
}

namespace utils {


namespace sync {
class Cond;

class Mutex
{
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

class MutexLocker {
  utils::sync::Mutex& m_mutex;

  public:
    inline MutexLocker(utils::sync::Mutex& mutex) : m_mutex(mutex)
    {
      m_mutex.lock();
    }

    inline ~MutexLocker() { m_mutex.unlock(); }
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

class Semaphore {
 public:
  Semaphore();
  virtual ~Semaphore();
  void wait();
  void signal();

 private:
  sem_t m_sem;
};

class Barrier {
  bool m_init;
  pthread_barrier_t m_barrier;

 public:
  Barrier();
  Barrier(unsigned int count);
  void init(unsigned int count);
  ~Barrier();
  void wait();
};

class CondBool {
  utils::sync::Cond m_cond;
  utils::sync::Mutex m_mutex;
  bool m_shouldlocked;

 public:
  CondBool();
  ~CondBool();
  void wait();
  void signal();
  void broadcast();
};
}

class AsyncQueue
{
  public:
    using Function = std::function<void(std::thread::id)>;

    AsyncQueue ();
    virtual ~AsyncQueue ();

    bool push(const Function& function);
    bool push(Function&& function);

    void stop();

    const std::thread* getThread() const;

  private:
    std::thread* m_thread = nullptr;
    std::condition_variable m_cv;
    std::mutex m_mutex;
    std::queue<Function> m_queue;
    std::atomic_bool m_stop;

    void runThread ();

    template<typename Function>
    bool _push(Function&& function)
    {
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_stop.load())
        {
          m_queue.push (std::forward<Function>(function));
          m_cv.notify_one ();
        }
        else
        {
          return false;
        }
      }
      runThread();
      return true;
    }
};

class Thread
{
  public:
    using Function = std::function<void(void*)>;

  protected:
    virtual void onRun (std::thread::id id);

  public:
    Thread ();
    virtual ~Thread ();

    void run (Function function, void* ptr);
    void stop();
    
    std::thread::id get_id() const
    {
      return m_asyncQueue.getThread()->get_id();
    }
  private:
    AsyncQueue m_asyncQueue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_onrunDone = false;
};
}
}

namespace std
{
  std::string to_string(const std::thread* thread);
  std::string to_string(std::thread::id id);
}

#endif
