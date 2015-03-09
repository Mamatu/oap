/* 
 * File:   ThreadsMapper.cpp
 * Author: mmatula
 * 
 * Created on February 20, 2014, 7:42 PM
 */

#include "ThreadsMapper.h"
#include <string.h>
#include <vector>
#include <math.h>

namespace utils {
namespace mapper {

void SetThreadsBlocks(uintt blocks[2], uintt threads[2], uintt w, uintt h,
    uintt threadsLimit) {
    uintt sqrtThreads = sqrt(threadsLimit);
    blocks[0] = 1;
    blocks[1] = 1;
    if (w * h < threadsLimit) {
        threads[0] = w;
        threads[1] = h;
    } else if (sqrtThreads <= w && sqrtThreads <= h) {
        threads[0] = sqrtThreads;
        threads[1] = sqrtThreads;
        while (threads[0] * blocks[0] < w) {
            ++blocks[0];
        }
        while (threads[1] * blocks[1] < h) {
            ++blocks[1];
        }
    } else {
        floatt factor = w / static_cast<floatt> (w + h);
        threadsLimit = 2 * sqrtThreads;
        threads[0] = (factor) * threadsLimit;
        threads[1] = (1 - factor) * threadsLimit;
        while (threads[0] * blocks[0] < w) {
            ++blocks[0];
        }
        while (threads[1] * blocks[1] < h) {
            ++blocks[1];
        }
    }
}

}

}