/*
 * Copyright 2016 Marcin Matula
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
#include "BacktraceUtils.h"

#define DEBUG
//#define DEBUG_MATRIX_OPERATIONS

#define STREAM stdout

//typedef void* LHandle;


#ifdef DEBUG
#include <stdio.h>
#endif

#define debug1(file, x, ...) fprintf(file, x, ##__VA_ARGS__); 


#ifdef DEBUG

#define debug(x, ...) fprintf(STREAM, x, ##__VA_ARGS__); fprintf(STREAM, "\n");

#define debugAssert(x) assert(x);

#define debugError(x, ...) fprintf(stderr, x, ##__VA_ARGS__); fprintf(stderr, "\n");

#define debugFunc() fprintf(STREAM, "%s %s : %d  \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugFuncBegin() fprintf(STREAM, "++ %s %s : %d  \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugFuncEnd() fprintf(STREAM, "-- %s %s : %d \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugPrintStack() BacktraceUtils::GetInstance().printBacktrace();

#define debugLongTest() debug("Test in progress. Please wait it can take some time...")

#else

#define debug( x, ...) 

#define debugAssert(x)

#define debugError(x, ...)

#define debugFunc() 

#define debugFuncBegin() 

#define debugFuncEnd() 

#define debugPrintStack()

#define debugLongTest()

#endif

#endif	/* TYPES_H */
