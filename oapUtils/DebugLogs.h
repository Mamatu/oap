#ifndef TYPES_H
#define	TYPES_H

#include <string.h>
#include <assert.h>
#include "LHandle.h"
#include "BacktraceUtils.h"

#define DEBUG
//#define DEBUG_MATRIX_OPERATIONS

#define STREAM stdout

//typedef void* LHandle;


#ifdef DEBUG
#include <stdio.h>
#endif

typedef unsigned int handle;

#define debug1(file, x, ...) fprintf(file, x, ##__VA_ARGS__); 


#ifdef DEBUG

#define debug(x, ...) fprintf(STREAM, x, ##__VA_ARGS__); fprintf(STREAM, "\n");

#define debugAssert(x) assert(x);

#define debugError(x, ...) fprintf(stderr, x, ##__VA_ARGS__); fprintf(stderr, "\n");

#define debugFunc() fprintf(STREAM, "%s %s : %d  \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugFuncBegin() fprintf(STREAM, "++ %s %s : %d  \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugFuncEnd() fprintf(STREAM, "-- %s %s : %d \n", __FUNCTION__,__FILE__,__LINE__); 

#define debugPrintStack() BacktraceUtils::GetInstance().printBacktrace();


#else

#define debug( x, ...) 

#define debugAssert(x)

#define debugError(x, ...)

#define debugFunc() 

#define debugFuncBegin() 

#define debugFuncEnd() 

#define debugPrintStack()

#endif

#endif	/* TYPES_H */

