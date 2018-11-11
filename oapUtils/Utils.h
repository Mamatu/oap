/*
 * Copyright 2016 - 2018 Marcin Matula
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

#ifndef UTILS_H
#define UTILS_H

#include "Dim3.h"

#include <math.h>

#include "Matrix.h"
#include "MatrixUtils.h"
#include "MatrixEx.h"

namespace utils {

class Compare {
 public:
  Compare();

  virtual ~Compare();

  virtual bool rule(const floatt& arg1, const floatt& arg2) = 0;

  bool compare(math::Matrix* matrix, floatt d);
};

math::Matrix* create(const math::Matrix& arg);

bool AlmostEquals(floatt a, floatt b);

bool AlmostEquals(floatt a, floatt b, floatt epsilon);

void diff(math::Matrix* output, math::Matrix* m1, math::Matrix* m2);

bool IsEqual(const math::Matrix& m1, const math::Matrix& m2,
             math::Matrix** diff = NULL);

bool HasValues(const math::Matrix& m1, const math::Matrix& m2,
               math::Matrix** diff = NULL);

bool IsIdentityMatrix(const math::Matrix& m1, math::Matrix** output);

bool IsDiagonalMatrix(const math::Matrix& m1, floatt value,
                      math::Matrix** output);

bool IsIdentityMatrix(const math::Matrix& m1);

bool IsDiagonalMatrix(const math::Matrix& m1, floatt value);

bool isEqual(const MatrixEx& matrixEx, const uintt* buffer);

bool areEqual(math::Matrix* matrix, floatt value);

bool areNotEqual(math::Matrix* matrix, floatt value);

typedef std::pair<size_t, size_t> Range;

template <typename T>
class Ranges : public std::vector<std::pair<T, Range> > {};

template <typename T>
bool isMatched(T* array, size_t length, const Ranges<T>& matched,
               const Ranges<T>& notmatched) {}

// template<typename T> typedef typename std::vector<std::pair<T, Range<T> > >
// Ranges<T>;

// template<typename T> bool isMatched(T* array, size_t length) {

//}

template <typename T>
T getSum(T* buffer, size_t length, T bzfactor = 1) {
  T output = 0;
  for (uintt fa = 0; fa < length; ++fa) {
    T v = buffer[fa];
    if (v < 0) {
      v = v * bzfactor;
    }
    output += v;
  }
  return output;
}

template <typename T>
T getMean(T* buffer, size_t length, T bzfactor) {
  T output = getSum(buffer, length, bzfactor);
  return output / length;
}

template <typename T>
std::pair<T, uintt> getLargest(T* buffer, size_t length) {
  T max = buffer[0];
  uintt index = 0;
  for (uintt fa = 1; fa < length; ++fa) {
    if (buffer[fa] > max) {
      max = buffer[fa];
      index = fa;
    }
  }
  return std::make_pair(max, index);
}

template <typename T>
std::pair<T, uintt> getSmallest(T* buffer, size_t length) {
  T min = buffer[0];
  uintt index = 0;
  for (uintt fa = 1; fa < length; ++fa) {
    if (buffer[fa] < min) {
      min = buffer[fa];
      index = fa;
    }
  }
  return std::make_pair(min, index);
}

#ifndef CUDA

inline std::string gridDimToStr(const dim3& dim3) {
  std::ostringstream s;
  s << "gridDim = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
  return s.str();
}

inline std::string blockDimToStr(const dim3& dim3) {
  std::ostringstream s;
  s << "blockDim = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
  return s.str();
}

inline std::string blockIdxToStr(const dim3& dim3) {
  std::ostringstream s;
  s << "blockIdx = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
  return s.str();
}

inline std::string threadIdxToStr(const dim3& dim3) {
  std::ostringstream s;
  s << "threadIdx = [" << dim3.x << ", " << dim3.y << ", " << dim3.z << "]";
  return s.str();
}

inline std::string cudaDimsToStr(const dim3& threadIdx) {
  std::string output = " ";
  output += threadIdxToStr(threadIdx) + " ";
  // output += blockIdxToStr(blockIdx) + " ";
  // output += blockDimToStr(blockDim) + " ";
  // output += gridDimToStr(gridDim) + " ";
  return output;
}
#endif
}

#endif /* UTILS_H */
