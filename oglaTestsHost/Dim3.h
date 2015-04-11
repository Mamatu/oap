/* 
 * File:   Dim3.h
 * Author: mmatula
 *
 * Created on March 10, 2015, 10:06 PM
 */

#ifndef DIM3_H
#define	DIM3_H

#include <cuda.h>

#ifdef CUDATEST

#include <stdlib.h>
#include <map>
#include "Math.h"
#include <pthread.h>

class Dim3 {
public:

    Dim3() {
        x = 0;
        y = 0;
        z = 1;
    }

    Dim3(size_t tuple[2]) {
        x = tuple[0];
        y = tuple[1];
        z = 1;
    }

    Dim3(uintt tuple[2]) {
        x = tuple[0];
        y = tuple[1];
        z = 1;
    }

    Dim3(size_t x, size_t y) {
        this->x = x;
        this->y = y;
        z = 1;
    }

    void clear() {
        x = 0;
        y = 0;
    }

    size_t x;
    size_t y;
    size_t z;
};

extern Dim3 blockIdx;
extern Dim3 blockDim;
extern Dim3 gridDim;

void ResetCudaCtx();

class ThreadIdx {
public:
    Dim3 threadIdx;

    void clear();

    typedef std::map<pthread_t, ThreadIdx> ThreadIdxs;

    static ThreadIdxs m_threadIdxs;
};

#endif

#endif	/* DIM3_H */

