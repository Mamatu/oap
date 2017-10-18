/*
 * Copyright 2016, 2017 Marcin Matula
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




#include "ThreadsMapper.h"
#include <string.h>
#include <vector>
#include <math.h>

namespace utils {

namespace mapper {

inline void increaseBlock(uint* blocks, uint* threads, int index,
                          uint limit) {
  while (threads[index] * blocks[index] < limit) {
    ++blocks[index];
  }
}

void SetThreadsBlocks(uint blocks[2], uint threads[2], uint w, uint h, uint threadsLimit) {
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
