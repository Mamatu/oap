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

#ifndef OAP_CU_KERNEL_OPERATIONS_MACROS_H
#define OAP_CU_KERNEL_OPERATIONS_MACROS_H

#include "CuModuloMacro.h"

/**
 * Convolution
 * Calculation of indecies:
 *  - threadIndexX, threadIndexY - index in cache
 *  - cacheIdx - composed from threadIndexX and threadIndexY
 *
 *  |O11 O12 O13|    |P11 P12 P13 P14|           |K11 K12|
 *  |O21 O22 O23| =  |P21 P22 P23 P24| convolve  |K21 K22|
 *  |O31 O32 O33|    |P31 P32 P33 P34|           
 *                   |P41 P42 P43 P44|
 *
 *  O11 = P11 * K11 + P12 * K12 + P21 * K21 + P22 * K22
 *  O12 = P12 * K11 + P13 * K12 + P22 * K21 + P23 * K22
 *
 * Cache:
 *  |P11*K11 P12*K12 P21*K21 P22*K22 P12*K11 P13*K12 P22*K21 P23*K22 P13*K11 P14*K12 P23*K21 P24*K22 ...|
 *  |P21*K11 P22*K12 P31*K21 P32*K22 P22*K11 P23*K12 P32*K21 P33*K22 P23*K11 P24*K12 P33*K21 P34*K22 ...|
 */
#define KEROPER_CONVOLUTION_CALCULATE_CACHE_COLUMNS(matrix, kernel, columns, rows) (matrix columns - kernel columns + 1) * kernel columns * kernel rows
#define KEROPER_CONVOLUTION_CALCULATE_CACHE_ROWS(matrix, kernel, rows) (matrix rows - kernel rows + 1)
#define KEROPER_CONVOLUTION_CALCULATE_CACHE_IDX(matrix, kernel, columns, rows) threadIndexX + threadIndexY * KEROPER_CONVOLUTION_CALCULATE_CACHE_COLUMNS(matrix, kernel, columns, rows)
#define KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_X(kernel, columns, rows) CU_MODULO(threadIndexX, kernel columns) + (threadIndexX / (kernel columns * kernel rows))
#define KEROPER_CONVOLUTION_CALCULATE_PARAM_IDX_Y(kernel, columns, rows) threadIndexY + (CU_MODULO(threadIndexX, kernel columns * kernel rows)) / (kernel columns)

/**
 * Pooling
 * Calculation of indecies:
 *  - threadIndexX, threadIndexY - index in cache
 *  - cacheIdx - composed from threadIndexX and threadIndexY
 *
 *  |O11 O12|    |P11 P12 P13 P14|           |K11 K12|
 *  |O21 O22| =  |P21 P22 P23 P24| convolve  |K21 K22|
 *               |P31 P32 P33 P34|           
 *               |P41 P42 P43 P44|
 *
 *  O11 = P11 * K11 + P12 * K12 + P21 * K21 + P22 * K22
 *  O12 = P13 * K11 + P14 * K12 + P23 * K21 + P24 * K22
 *
 * Cache:
 *  |P11*K11 P12*K12 P21*K21 P22*K22 P13*K11 P14*K12 P23*K21 P24*K22|
 *  |P31*K11 P32*K12 P41*K21 P42*K22 P33*K11 P34*K12 P43*K21 P44*K22|
 */
#define KEROPER_POOLING_CALCULATE_CACHE_COLUMNS(matrix, kernel, columns, rows) (matrix columns / kernel columns) * (kernel columns * kernel rows)
#define KEROPER_POOLING_CALCULATE_CACHE_ROWS(matrix, kernel, rows) (matrix rows / kernel rows)
#define KEROPER_POOLING_CALCULATE_CACHE_IDX(matrix, kernel, columns, rows) threadIndexX + threadIndexY * KEROPER_POOLING_CALCULATE_CACHE_COLUMNS(matrix, kernel, columns, rows)
#define KEROPER_POOLING_CALCULATE_PARAM_IDX_X(kernel, columns, rows)  CU_MODULO(threadIndexX, kernel columns) + ((threadIndexX / (kernel columns * kernel rows)) * kernel columns)
#define KEROPER_POOLING_CALCULATE_PARAM_IDX_Y(kernel, columns, rows) (threadIndexY * kernel rows) + (CU_MODULO(threadIndexX, (kernel columns * kernel rows)) / kernel columns)

#define KEROPER_CALCULATE_KERNEL_IDX(kernel, columns, rows) CU_MODULO(threadIndexX, kernel columns * kernel rows)

#define KEROPER_CALCULATE_OUTPUT_DIM(matrix, kernel, dim) (matrix dim - kernel dim + 1)

#define KEROPER_CALCULATE_OUTPUT_IDX_X(kernel, columns, rows) threadIndexX / (kernel columns * kernel rows)

#define KEROPER_IS_OUTPUT_IDX(kernel, columns, rows) (CU_MODULO(cacheIdx, kernel columns * kernel rows) == 0)

#define KEROPER_CACHE_CODE(type, params, kernel, cache, columns, rows, cache_set_code)            \
  uintt cacheW = KEROPER_##type##_CALCULATE_CACHE_COLUMNS (params, kernel, columns, rows);        \
  uintt cacheH = KEROPER_##type##_CALCULATE_CACHE_ROWS (params, kernel, rows);                    \
  uintt kidx = KEROPER_CALCULATE_KERNEL_IDX(kernel, columns, rows);                               \
  uintt px = KEROPER_##type##_CALCULATE_PARAM_IDX_X(kernel, columns, rows);                       \
  uintt py = KEROPER_##type##_CALCULATE_PARAM_IDX_Y(kernel, columns, rows);                       \
  const uintt cacheIdx = KEROPER_##type##_CALCULATE_CACHE_IDX (params, kernel, columns, rows);    \
  cache[cacheIdx] = cache_set_code;
  //cache[cacheIdx] = GetRe (params0, px, py) * GetReIndex (kernel, kidx);

#endif
