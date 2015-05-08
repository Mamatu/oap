/*
 * File:   mutex.h
 * Author: marcin
 *
 * Created on 26 December 2012, 20:38
 */

#ifndef OGLA_THREAD_UTILS_H
#define OGLA_THREAD_UTILS_H

#include <pthread.h>
#include <semaphore.h>
#include "DebugLogs.h"

namespace utils {

namespace sync {
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

class MutexLocker {
  utils::sync::Mutex& m_mutex;

 public:
  inline MutexLocker(utils::sync::Mutex& mutex) : m_mutex(mutex) {
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
  Barrier(int count);
  void init(int count);
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

typedef void (*ThreadFunction_f)(void* ptr);

class Thread {
  ThreadFunction_f m_function;
  void* m_ptr;
  pthread_t m_thread;
  static void* Execute(void* m_ptr);

  utils::sync::Cond m_cond;
  utils::sync::Mutex m_mutex;
  bool m_iscond;
  bool m_isonethread;

 protected:
  virtual void onRun(pthread_t threadId);

 public:
  Thread();
  virtual ~Thread();
  void setFunction(ThreadFunction_f _function, void* _ptr);
  void run(bool inTheSameThreead = false);
  void yield();
};
}
#endif
