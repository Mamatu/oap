/* 
 * File:   ThreadsCpu.cpp
 * Author: mmatula
 * 
 * Created on August 2, 2014, 9:40 PM
 */

#include "ThreadsCpu.h"

ThreadsCountProperty::ThreadsCountProperty() : m_threadsCount(1) {
}

ThreadsCountProperty::~ThreadsCountProperty() {
}

void ThreadsCountProperty::setThreadsCount(uintt threadsCount) {
    m_threadsCount = threadsCount;
}
