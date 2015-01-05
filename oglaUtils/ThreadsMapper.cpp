/* 
 * File:   ThreadsMapper.cpp
 * Author: mmatula
 * 
 * Created on February 20, 2014, 7:42 PM
 */

#include "ThreadsMapper.h"
#include <string.h>
#include <vector>

namespace utils {
namespace mapper {

void SetThreadsBlocks(uintt blocks[2], uintt threads[2], uintt w, uintt h,
    uintt threadsLimit, uintt factor) {
    threads[0] = w;
    threads[1] = h;
    blocks[0] = 1;
    blocks[1] = 1;
    int index = 0;
    while (threads[0] * threads[1] > threadsLimit) {
        threads[index] = threads[index] / factor;
        blocks[index] = blocks[index] * factor;
        index = index == 0 ? 1 : 0;
    }
}
}
}