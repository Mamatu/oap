/*
 * File:   Utils.h
 * Author: mmatula
 *
 * Created on March 22, 2015, 11:23 AM
 */

#ifndef UTILS_H
#define UTILS_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Dim3.h"

#include "Matrix.h"
#include "HostMatrixModules.h"
#include "HostMatrixModules.h"
#include "MatrixUtils.h"
#include "MatrixEx.h"

namespace utils {

class Compare {
 public:
  Compare() {}

  virtual ~Compare() {}

  virtual bool rule(const floatt& arg1, const floatt& arg2) = 0;

  inline bool compare(math::Matrix* matrix, floatt d) {
    if (NULL == matrix) {
      return false;
    }
    uintt length = matrix->rows * matrix->columns;
    for (uintt fa = 0; fa < length; ++fa) {
      if (!rule(matrix->reValues[fa], d)) {
        return false;
      }
    }
    return true;
  }
};

inline math::Matrix* create(const math::Matrix& arg) {
  return host::NewMatrix(arg.reValues != NULL, arg.imValues != NULL,
                         arg.columns, arg.rows);
}

inline bool AlmostEquals(floatt a, floatt b) {
  return fabs(a - b) < std::numeric_limits<floatt>::epsilon();
}

inline bool AlmostEquals(floatt a, floatt b, floatt epsilon) {
  return fabs(a - b) < epsilon;
}

inline void diff(math::Matrix* output, math::Matrix* m1, math::Matrix* m2) {
  for (uintt fa = 0; fa < output->columns; ++fa) {
    for (uintt fb = 0; fb < output->rows; ++fb) {
      uintt index = fa + fb * m1->columns;
      if (output->reValues != NULL) {
        output->reValues[index] = m1->reValues[index] - m2->reValues[index];
      }
      if (output->imValues != NULL) {
        output->imValues[index] = m1->imValues[index] - m2->imValues[index];
      }
    }
  }
}

inline bool IsEqual(const math::Matrix& m1, const math::Matrix& m2,
                    math::Matrix** output) {
  (*output) = NULL;
  if (m1.columns != m2.columns || m1.rows != m2.rows) {
    return false;
  }
  bool status = true;
  for (uintt fa = 0; fa < m1.columns; ++fa) {
    for (uintt fb = 0; fb < m1.rows; ++fb) {
      uintt index = fa + fb * m1.columns;
      if (m1.reValues != NULL && m2.reValues != NULL) {
        floatt re1 = (m1.reValues[index]);
        floatt re2 = (m2.reValues[index]);

        if (!AlmostEquals(re1, re2, 0.0001)) {
          status = false;
          if (*output == NULL) {
            (*output) = create(m1);
          }
          (*output)->reValues[index] = m1.reValues[index] - m2.reValues[index];
        }
      }
      if (m1.imValues != NULL && m2.imValues != NULL) {
        floatt im1 = (m1.imValues[index]);
        floatt im2 = (m2.imValues[index]);

        if (!AlmostEquals(im1, im2, 0.001)) {
          status = false;
          if (*output == NULL) {
            (*output) = create(m1);
          }
          (*output)->imValues[index] = m1.imValues[index] - m2.imValues[index];
        }
      }
    }
  }
  return status;
}

inline bool IsIdentityMatrix(const math::Matrix& m1, math::Matrix** output) {
  math::Matrix* matrix = host::NewMatrix(&m1);
  host::SetIdentity(matrix);
  bool isequal = IsEqual(m1, *matrix, output);
  host::DeleteMatrix(matrix);
  return isequal;
}

inline bool IsDiagonalMatrix(const math::Matrix& m1, floatt value,
                             math::Matrix** output) {
  math::Matrix* matrix = host::NewMatrix(&m1);
  host::SetDiagonalMatrix(matrix, value);
  bool isequal = IsEqual(m1, *matrix, output);
  host::DeleteMatrix(matrix);
  return isequal;
}

inline bool IsIdentityMatrix(const math::Matrix& m1) {
  return IsIdentityMatrix(m1, NULL);
}

inline bool IsDiagonalMatrix(const math::Matrix& m1, floatt value) {
  return IsDiagonalMatrix(m1, value, NULL);
}

inline bool isEqual(const MatrixEx& matrixEx, const uintt* buffer) {
  if (matrixEx.bcolumn == buffer[0] && matrixEx.ecolumn == buffer[1] &&
      matrixEx.brow == buffer[2] && matrixEx.erow == buffer[3] &&
      matrixEx.boffset == buffer[4] && matrixEx.eoffset == buffer[5]) {
    return true;
  }
  return false;
}

inline bool areEqual(math::Matrix* matrix, int d) {
  class CompareImpl : public Compare {
   public:
    bool rule(const floatt& arg1, const floatt& arg2) { return arg1 == arg2; }
  };
  CompareImpl compareImpl;
  return compareImpl.compare(matrix, d);
}

inline bool areNotEqual(math::Matrix* matrix, int d) {
  class CompareImpl : public Compare {
   public:
    bool rule(const floatt& arg1, const floatt& arg2) { return arg1 != arg2; }
  };
  CompareImpl compareImpl;
  return compareImpl.compare(matrix, d);
}

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
  output += blockIdxToStr(blockIdx) + " ";
  output += blockDimToStr(blockDim) + " ";
  output += gridDimToStr(gridDim) + " ";
  return output;
}
#endif
}

#endif /* UTILS_H */
