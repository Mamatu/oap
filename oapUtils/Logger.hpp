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

// This code is inspired by https://github.com/dmcrodrigues/macro-logger/blob/master/macrologger.h

#ifndef LOGGER_H
#define	LOGGER_H

#include <string.h>
#include <typeinfo>
#include <stdexcept>
#include <sstream>
#include <iostream>

#include "oapAssertion.hpp"

#define STREAM stdout

#include <stdio.h>

#define OPEN_MARK "+ "
#define CLOSE_MARK "- "
#define LOG_ARGS() __FILE__, __FUNCTION__, __LINE__

#define NO_LOG          0x00
#define ERROR_LEVEL     0x01
#define INFO_LEVEL      0x02
#define DEBUG_LEVEL     0x03
#define TRACE_LEVEL     0x04

#ifndef LOG_LEVEL
#define LOG_LEVEL DEBUG_LEVEL
#endif

#define logIntoStdout(x, ...) fprintf(stdout, x, ##__VA_ARGS__);
#define logIntoStderr(x, ...) fprintf(stderr, x, ##__VA_ARGS__);

#define LOCATION_FORMAT "%s %s %d "
#define LOCATION __FUNCTION__ __FILE__ __LINE__

#define ENDL "\n"

#if LOG_LEVEL >= ERROR_LEVEL
#define logError(x, ...) logIntoStderr (LOCATION_FORMAT x ENDL, LOG_ARGS(), ##__VA_ARGS__);
#else
#define logError(x, ...)
#endif

#if LOG_LEVEL >= INFO_LEVEL
#define logInfo(x, ...) logIntoStdout (LOCATION_FORMAT x ENDL, LOG_ARGS(), ##__VA_ARGS__);
#else
#define logInfo(x, ...)
#endif

#if LOG_LEVEL >= DEBUG_LEVEL
#define logDebug(x, ...) logIntoStdout (LOCATION_FORMAT x ENDL, LOG_ARGS(), ##__VA_ARGS__);
#else
#define logDebug(x, ...)
#endif

#if LOG_LEVEL >= TRACE_LEVEL

#define logTraceS(x, ...) logIntoStdout (OPEN_MARK LOCATION_FORMAT x ENDL, LOG_ARGS(), ##__VA_ARGS__);
#define logTraceE(x, ...) logIntoStdout (CLOSE_MARK LOCATION_FORMAT x ENDL, LOG_ARGS(), ##__VA_ARGS__);

class LogTraceE final
{
    const char* m_func;
    const char* m_file;
    int m_line;
  public:
    LogTraceE (const char* func, const char* file, int line) :
      m_func(func), m_file(file), m_line(line)
    {
    }

    ~LogTraceE ()
    {
      logTraceE("%s %s %d", m_func, m_file, m_line);
    }
};

#define logTrace(x, ...) logIntoStdout (LOCATION_FORMAT x ENDL, LOG_ARGS(), ##__VA_ARGS__);
#define LOG_TRACE(x, ...) logTraceS (x, ##__VA_ARGS__); LogTraceE logTraceObj (__FUNCTION__,__FILE__,__LINE__);
#else
#define logTrace(x, ...)
#define logTraceS(x, ...)
#define logTraceE(x, ...)
#define LOG_TRACE(x, ...)
#endif

#define debugFunc() logDebug("");
#define traceFunction() logTrace("")

#define debugFuncBegin() logDebug("%s", "+");
#define debugFuncEnd() logDebug("%s" , "-");

#define debug(x, ...) logDebug(x, ##__VA_ARGS__);
#define debugInfo(x, ...) logDebug(x, ##__VA_ARGS__);
#define debugError(x, ...) logError(x, ##__VA_ARGS__);

#define debugAssertMsg(x, msg, ...) if (!(x)) { logError ("ASSERTION\n"); logError(msg, ##__VA_ARGS__); debugAssert(x); }
#define debugExceptionMsg(x, msg, ...) if (!(x)) { logError ("ASSERTION\n"); logError(msg, ##__VA_ARGS__); debugAssert(x); }

#define logInfoLongTest() logInfo("Test in progress. Please wait it can take some time...")

inline void throwExceptionMsg(bool x, const std::string& str) { if (!(x)) { std::cout << str << std::endl; throw std::runtime_error(str); } }
inline void throwExceptionMsg(bool x, const std::stringstream& stream) { throwExceptionMsg(x, stream.str()); }

#define debugException(ex) logDebug("Exception: %s %s", typeid(ex).name(), ex.what())


#endif	/* TYPES_H */
