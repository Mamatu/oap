/* 
 * File:   BacktraceUtils.cpp
 * Author: mmatula
 * 
 * Created on January 11, 2015, 10:14 PM
 */

#include "BacktraceUtils.h"
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

BacktraceUtils::BacktraceUtils() {
    m_size = 0;
    m_buffer = NULL;
    m_ptrsCount = 0;
}

BacktraceUtils::BacktraceUtils(const BacktraceUtils& orig) {
}

BacktraceUtils::~BacktraceUtils() {
}

BacktraceUtils* BacktraceUtils::m_backtraceUtils = NULL;

BacktraceUtils& BacktraceUtils::GetInstance() {
    if (BacktraceUtils::m_backtraceUtils == NULL) {
        BacktraceUtils::m_backtraceUtils = new BacktraceUtils();
    }
    return *(BacktraceUtils::m_backtraceUtils);
}

char** BacktraceUtils::repeatReadBacktrace() {
    m_size = m_size * 2;
    delete[] m_buffer;
    m_buffer = new void*[m_size];
    m_ptrsCount = backtrace(m_buffer, m_size);
    m_strings = backtrace_symbols(m_buffer, m_ptrsCount);
    if (m_strings == NULL) {
        return repeatReadBacktrace();
    }
    return m_strings;
}

const std::vector<std::string>& BacktraceUtils::readBacktrace(bool showThisObjectFrames) {
    if (m_buffer == NULL) {
        m_size = 1024;
        m_buffer = new void*[m_size];
    }
    m_ptrsCount = backtrace(m_buffer, m_size);
    m_strings = backtrace_symbols(m_buffer, m_ptrsCount);
    if (m_strings == NULL) {
        m_strings = repeatReadBacktrace();
    }
    m_backtrace.clear();
    for (int fa = 0; fa < m_ptrsCount; fa++) {
        std::string line = m_strings[fa];
        if (showThisObjectFrames == false
            && line.find("BacktraceUtils") == std::string::npos) {
            m_backtrace.push_back(line);
        }
    }
    free(m_strings);
    return m_backtrace;
}

void BacktraceUtils::printBacktrace(bool showThisObjectFrames) {
    readBacktrace(showThisObjectFrames);
    for (Backtrace::const_iterator it = m_backtrace.begin();
        it != m_backtrace.end(); ++it) {
        printf("%s \n", it->c_str());
    }
}
