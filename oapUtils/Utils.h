
#ifndef UTILS_H
#define UTILS_H

#include "Dim3.h"

#include "Matrix.h"
#include "HostMatrixModules.h"
#include "HostMatrixModules.h"
#include "MatrixUtils.h"
#include "MatrixEx.h"
#include <math.h>

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

bool IsIdentityMatrix(const math::Matrix& m1, math::Matrix** output);

bool IsDiagonalMatrix(const math::Matrix& m1, floatt value,
                      math::Matrix** output);

bool IsIdentityMatrix(const math::Matrix& m1);

bool IsDiagonalMatrix(const math::Matrix& m1, floatt value);

bool isEqual(const MatrixEx& matrixEx, const uintt* buffer);

bool areEqual(math::Matrix* matrix, int d);

bool areNotEqual(math::Matrix* matrix, int d);

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
T getSum(T* buffer, size_t length) {
  T output = 0;
  for (uintt fa = 0; fa < length; ++fa) {
    output += buffer[fa];
  }
  return output;
}

template <typename T>
T getMean(T* buffer, size_t length) {
  T output = getSum(buffer, length);
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
  return std::make_pair<T, uintt>(max, index);
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
  return std::make_pair<T, uintt>(min, index);
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
