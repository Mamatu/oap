/* 
 * File:   ThreadsMapper.h
 * Author: mmatula
 *
 * Created on February 20, 2014, 7:42 PM
 */

#ifndef THREADSMAPPER_H
#define	THREADSMAPPER_H

#include <stdio.h>
#include "DebugLogs.h"
#include "Math.h"

namespace utils {

namespace mapper {

void SetThreadsBlocks(uintt blocks[2], uintt threads[2], 
    uintt w, uintt h, uintt threadsLimit);

template<typename T> struct ThreadsMap {
    T beginColumn;
    T endColumn;
    T beginRow;
    T endRow;
};

template<typename T> T* allocMap(T threadsCount) {
    return new T[threadsCount * 4];
}

template<typename T> void freeMap(T*& buffer) {
    delete[] buffer;
    buffer = NULL;
}

template<typename T> void getThreadsMap(ThreadsMap<T>& threadsMap,
    T* array, T threadIndex) {
    memcpy(&threadsMap, array + threadIndex * 4, sizeof (ThreadsMap<T>));
}

template<typename T> void createThreadsMap(T* array,
    T blocksCount[2], T threadsCount[2],
    T width, T height) {
    debugAssert(threadsCount[0] != 0);
    debugAssert(blocksCount[0] != 0);
    debugAssert(blocksCount[1] != 0);
    debugAssert(threadsCount[1] != 0);
    T threadsCountX = blocksCount[0] * threadsCount[0];
    T threadsCountY = blocksCount[1] * threadsCount[1];
    T widthPT = width / threadsCountX;
    T heightPT = height / threadsCountY;
    T bc = 0;
    T br = 0;
    if (NULL != array) {
        for (T fa = 0; fa < threadsCountX; fa++) {
            for (T fb = 0; fb < threadsCountY; fb++) {
                if (bc >= width) {
                    bc = 0;
                }
                array[(fa + threadsCountX * fb) * 4] = bc;
                if (bc + widthPT <= width) {
                    bc = bc + widthPT;
                }
                array[(fa + threadsCountX * fb) * 4 + 1] = bc;
                if (fa == threadsCountX - 1) {
                    array[(fa + threadsCountX * fb) * 4 + 1] = width;
                }
                if (br >= height) {
                    br = 0;
                }
                array[(fa + threadsCountX * fb) * 4 + 2] = br;
                if (br + heightPT <= height) {
                    br = br + heightPT;
                }
                array[(fa + threadsCountX * fb) * 4 + 3] = br;
                if (fb == threadsCountY - 1) {
                    array[(fa + threadsCountX * fb) * 4 + 3] = height;
                }
            }
        }
    }
}

template<typename T> T getThreadsCount(T threadsCount,
    T width, T height) {
    T blocksCount[2] = {1, 1};
    T threadCount1[2] = {threadsCount, 1};
    if (threadsCount <= width || threadsCount <= height) {
        createThreadsMap(NULL, blocksCount, threadCount1, width, height);
    } else {
        threadsCount = createThreadsMap(NULL, threadCount1[0] - 1, width, height);
    }
    return threadCount1[0];
}

template<typename T> T createThreadsMap(T* array,
    T threadsCount,
    T width, T height) {
    memset(array, 0, sizeof (T) * threadsCount * 4);
    T blocksCount[2] = {1, 1};
    T threadCount1[2] = {threadsCount, 1};
    if ((threadsCount <= width && threadsCount <= height) ||
        threadsCount <= width) {
        createThreadsMap(array, blocksCount, threadCount1, width, height);
    } else if (threadsCount <= height) {
        threadCount1[0] = 1;
        threadCount1[1] = threadsCount;
        createThreadsMap(array, blocksCount, threadCount1, width, height);
    } else {
        threadsCount = createThreadsMap(array, threadsCount - 1, width, height);
    }
    return threadsCount;
}

template<typename T> void createThreadsMap(T* array,
    T threadsCount[2],
    T width, T height) {
    T blocksCount[2] = {1, 1};
    createThreadsMap(array, blocksCount, threadsCount, width, height);
}

template<typename T > T createThreadsMap(T* array,
    T threadsCount,
    T length) {
    T a = T(1);
    return createThreadsMap(array, threadsCount, length, a);
}
}
}
#endif	/* THREADSMAPPER_H */

