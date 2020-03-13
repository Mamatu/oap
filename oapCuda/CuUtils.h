/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <stdio.h>
#include "Math.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include "CuCore.h"

#ifndef DEBUG

#define cuda_debug_buffer(buffer, len)
#define cuda_debug_matrix_ex(s, mo)
#define cuda_debug(x, ...)
#define cuda_debug_function()
#define cuda_debug_thread(tx, ty, arg, ...)

#else

#define IS_FIRST_THREAD                            \
  ((blockIdx.x * blockDim.x + threadIdx.x) == 0 && \
   (blockIdx.y * blockDim.y + threadIdx.y) == 0)

#define IS_THREAD(tx,ty) (blockIdx.x * blockDim.x + threadIdx.x) == tx && (blockIdx.y * blockDim.y + threadIdx.y) == ty

__hostdevice__ void CUDA_PrintBuffer(const char* s, floatt* buffer, uint length) {
  printf("%s = {", s);
  for (uint fa = 0; fa < length; ++fa) {
    printf("%f ", buffer[fa]);
  }
  printf("}\n");
}

__hostdevice__ void CUDA_PrintMatrix (const char* str, math::Matrix* m)
{
  printf ("%s = [\n", str);
  for (uintt fb = 0; fb < gRows (m); ++fb)
  {
    printf("[");
    for (uintt fa = 0; fa < gColumns (m); ++fa)
    {
      printf ("%f ", gReValues (m)[fb * gColumns (m) + fa]);
    }
    printf("]\n");
  }
  printf ("]\n");
}

__hostdevice__ void CUDA_PrintBufferUintt(uintt* buffer, uint length) {
  for (uint fa = 0; fa < length; ++fa) {
    printf("buffer[%u] = %u \n", fa, buffer[fa]);
  }
}

__hostdevice__ void CUDA_PrintBufferInt(int* buffer, uint length) {
  for (uint fa = 0; fa < length; ++fa) {
    printf("buffer[%u] = %d \n", fa, buffer[fa]);
  }
}

__hostdevice__ void CUDA_PrintFloat(floatt v) {
  printf("[%f]", v);
  printf("\n");
}

__hostdevice__ void CUDA_PrintInt(uint v) {
  printf("[%u]", v);
  printf("\n");
}

#define cuda_debug_matrix_ex(s, mo) \
  {                                 \
    if (IS_FIRST_THREAD) {          \
      printf("%s = \n", s);         \
      CUDA_PrintMatrixEx(mo);       \
    }                               \
    threads_sync();                 \
  }

#define cuda_debug_buffer(s, buffer, len) \
  {                                       \
    if (IS_FIRST_THREAD) {                \
      CUDA_PrintBuffer(s, buffer, len);      \
    }                                     \
    threads_sync();                 \
  }

#define cuda_debug_matrix(s, matrix) \
  {                                       \
    if (IS_FIRST_THREAD) {                \
      CUDA_PrintMatrix(s, matrix);      \
    }                                     \
    threads_sync();                 \
  }

#define cuda_debug_buffer_uint(s, buffer, len) \
  {                                            \
    if (IS_FIRST_THREAD) {                     \
      printf("%s = \n", s);                    \
      CUDA_PrintBufferUintt(buffer, len);      \
    }                                          \
  }

#define cuda_debug_buffer_int(s, buffer, len) \
  {                                           \
    if (IS_FIRST_THREAD) {                    \
      \ printf("%s = \n", s);                 \
      CUDA_PrintBufferInt(buffer, len);       \
    }                                         \
  }

#define cuda_debug(arg, ...)                                          \
  {                                                                   \
    if (IS_FIRST_THREAD) {                                            \
      printf("%s %s %d " arg " \n", __FUNCTION__, __FILE__, __LINE__, \
             ##__VA_ARGS__);                                          \
    }                                                                 \
  }

#define cuda_debug_abs(arg, ...)  \
  {                               \
    if (IS_FIRST_THREAD) {        \
      printf(arg, ##__VA_ARGS__); \
      printf("\n");               \
    }                             \
  }

#define cuda_debug_function()                                  \
  {                                                            \
    if (IS_FIRST_THREAD) {                                     \
      printf("%s %s %d \n", __FUNCTION__, __FILE__, __LINE__); \
    }                                                          \
  }

#define cuda_debug_thread(tx, ty, arg, ...)                                             \
  {                                                                                     \
    if (IS_THREAD(tx,ty)) {                                                             \
      printf("%s %s %d Thread: %u %u: " arg "\n", __FUNCTION__, __FILE__, __LINE__, tx, \
             ty, ##__VA_ARGS__);                                                        \
    }                                                                                   \
  }

#endif

#endif /* CUUTILS_H */
