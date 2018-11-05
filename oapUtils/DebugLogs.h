/*
 * Copyright 2016 - 2018 Marcin Matula
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



#ifndef TYPES_H
#define	TYPES_H

#include <string.h>
#include <assert.h>
#include <typeinfo>
#include <stdexcept>
#include <sstream>
#include <iostream>

#include "TraceLog.h"

#define STREAM stdout

#ifdef DEBUG

#include <stdio.h>

#define debug(x, ...) fprintf(STREAM, "%s %s : %d  ", __FUNCTION__,__FILE__,__LINE__);  fprintf(STREAM, x, ##__VA_ARGS__); fprintf(STREAM, "\n");

#define debugInfo(x, ...) fprintf(STREAM, x, ##__VA_ARGS__); fprintf(STREAM, "\n");

#define debugAssert(x) assert(x);

#define debugAssertMsg(x, msg, ...) if (!(x)) { debug(msg, ##__VA_ARGS__); debugAssert(x); }

inline void debugExceptionMsg(bool x, const std::string& str) { if (!(x)) { std::cout << str << std::endl; throw std::runtime_error(str); } }
inline void debugExceptionMsg(bool x, const std::stringstream& stream) { debugExceptionMsg(x, stream.str()); }

#define debugError(x, ...) fprintf(stderr, "ERROR: "); fprintf(stderr, x, ##__VA_ARGS__); fprintf(stderr, "\n");

#define debugFunc() fprintf(STREAM, "%s %s : %d  \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugFuncBegin() fprintf(STREAM, "++ %s %s : %d  \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugFuncEnd() fprintf(STREAM, "-- %s %s : %d \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugPrintStack() BacktraceUtils::GetInstance().printBacktrace();

#define debugLongTest() debug("Test in progress. Please wait it can take some time...")

#define debugException(ex) debug("Exception: %s %s", typeid(ex).name(), ex.what())

#define initTraceBuffer(size) trace::InitTraceBuffer(size)

#define traceFunction() debugFunc() trace::Trace("%s %s %d\n", __FUNCTION__, __FILE__, __LINE__);

#define getTraceOutput(out)  trace::GetOutputString(out);

#else

#define debug( x, ...) 

#define debugInfo(x, ...)

#define debugAssert(x)

#define debugAssertMsg(x, msg, ...)

inline void debugExceptionMsg(bool, const std::stringstream&) {}
inline void debugExceptionMsg(bool, const std::string&) {}

#define debugError(x, ...)

#define debugFunc() 

#define debugFuncBegin() 

#define debugFuncEnd() 

#define debugPrintStack()

#define debugLongTest()

#define debugException(ex) debug("Exception: %s %s", typeid(ex).name(), ex.getMessage().c_str())

#define initTraceBuffer(size)

#define traceFunction()

#define getTraceOutput(out)

#endif




#endif	/* TYPES_H */
