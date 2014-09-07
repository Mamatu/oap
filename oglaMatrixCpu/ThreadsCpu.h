/* 
 * File:   ThreadsCpu.h
 * Author: mmatula
 *
 * Created on August 2, 2014, 9:40 PM
 */

#ifndef OGLA_THREADSCPU_H
#define	OGLA_THREADSCPU_H

#include "ThreadUtils.h"
#include "HostMatrixModules.h"
#include "Internal.h"

class ThreadsCountProperty {
protected:
    uintt m_threadsCount;
public:
    ThreadsCountProperty();
    virtual ~ThreadsCountProperty();
    virtual void setThreadsCount(uintt threadsCount);
};

template<typename T> class ThreadsCPU :
public ThreadsCountProperty {
    uintt* m_bmap;
protected:
    ThreadData<T>* m_threadData;
    uintt* getBMap() const;
public:
    ThreadsCPU();
    virtual ~ThreadsCPU();
    void setThreadsCount(uintt threadsCount);
};

template<typename T> ThreadsCPU<T>::ThreadsCPU() : ThreadsCountProperty(),
m_bmap(NULL), m_threadData(NULL) {
    m_threadData = new ThreadData<T>[m_threadsCount];
    m_bmap = utils::mapper::allocMap(m_threadsCount);
}

template<typename T> ThreadsCPU<T>::~ThreadsCPU() {
    if (m_bmap) {
        utils::mapper::freeMap(m_bmap);
        delete[] m_threadData;
    }
}

template<typename T> uintt* ThreadsCPU<T>::getBMap() const {
    return this->m_bmap;
}

template<typename T> void ThreadsCPU<T>::setThreadsCount(uintt threadsCount) {
    if (this->m_threadsCount < threadsCount || m_bmap == NULL) {
        if (m_bmap) {
            utils::mapper::freeMap(m_bmap);
            delete[] m_threadData;
            m_bmap = NULL;
        }
        if (m_bmap == NULL) {
            m_threadData = new ThreadData<T>[threadsCount];
            m_bmap = utils::mapper::allocMap(threadsCount);
        }
    }
    this->m_threadsCount = threadsCount;
}


#endif	/* THREADSCPU_H */

