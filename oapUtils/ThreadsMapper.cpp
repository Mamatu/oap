
#include "ThreadsMapper.h"
#include <string.h>
#include <vector>
#include <math.h>

namespace utils {

namespace mapper {

inline void increaseBlock(uintt* blocks, uintt* threads, int index,
                          uintt limit) {
  while (threads[index] * blocks[index] < limit) {
    ++blocks[index];
  }
}

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
    increaseBlock(blocks, threads, 0, w);
    increaseBlock(blocks, threads, 1, h);
  } else if (sqrtThreads <= w && sqrtThreads > h) {
    threads[0] = sqrtThreads;
    threads[1] = h;
    increaseBlock(blocks, threads, 0, w);
  } else if (sqrtThreads <= h && sqrtThreads > w) {
    threads[0] = w;
    threads[1] = sqrtThreads;
    increaseBlock(blocks, threads, 1, h);
  } else {
    floatt factor = w / static_cast<floatt>(w + h);
    threadsLimit = 2 * sqrtThreads;
    threads[0] = (factor)*threadsLimit;
    threads[1] = (1 - factor) * threadsLimit;

    increaseBlock(blocks, threads, 0, w);
    increaseBlock(blocks, threads, 1, h);
  }
}
}
}
