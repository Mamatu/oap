
#include "ThreadsCpu.h"

ThreadsCountProperty::ThreadsCountProperty() : m_threadsCount(1) {
}

ThreadsCountProperty::~ThreadsCountProperty() {
}

void ThreadsCountProperty::setThreadsCount(uintt threadsCount) {
    m_threadsCount = threadsCount;
}
