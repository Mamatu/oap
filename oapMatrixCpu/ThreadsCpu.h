/*
 * Copyright 2016, 2017 Marcin Matula
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




#ifndef OAP_THREADSCPU_H
#define	OAP_THREADSCPU_H

#include "ThreadUtils.h"
#include "HostMatrixUtils.h"
#include "ThreadData.h"

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
