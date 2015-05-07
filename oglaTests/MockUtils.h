/*
 * File:   matrixEq.h
 * Author: mmatula
 *
 * Created on December 23, 2014, 10:30 AM
 */

#ifndef MOCKUTILS_H
#define MOCKUTILS_H

#include "gtest/gtest.h"
//#include "gtest/internal/gtest-internal.h"
#include "gmock/gmock.h"
#include <limits>

#include "Utils.h"
#include "MatchersImpl.h"
#include "HostMatrixModules.h"

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
            (*output)->reValues[index] =
                fabs(m1.reValues[index] - m2.reValues[index]);
          }
        }
      }
      if (m1.imValues != NULL && m2.imValues != NULL) {
        floatt im1 = (m1.imValues[index]);
        floatt im2 = (m2.imValues[index]);

        if (!AlmostEquals(im1, im2, 0.0001)) {
          status = false;
          if (*output == NULL) {
            (*output) = create(m1);
            (*output)->imValues[index] =
                fabs(m1.imValues[index] - m2.imValues[index]);
          }
        }
      }
    }
  }
  return status;
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

MATCHER_P(MatrixIsEqual, matrix1, "") {
  math::Matrix* pmatrix = NULL;
  bool isequal = IsEqual((*arg), (*matrix1), &pmatrix);
  if (!isequal) {
    host::PrintMatrix("", pmatrix);
    host::DeleteMatrix(pmatrix);
  }
  return isequal;
}

inline Matcher<math::Matrix*> MatrixValuesAreEqual(floatt value) {
  return MakeMatcher(new MatrixValuesAreEqualMatcher(value));
}

inline Matcher<floatt> IsEqualSum(floatt* buffer, size_t length,
                                  const std::string& extra = "") {
  return MakeMatcher(
      new BufferSumIsEqualMatcher<floatt>(buffer, length, extra));
}

inline Matcher<uintt> IsEqualSum(uintt* buffer, size_t length,
                                 const std::string& extra = "") {
  return MakeMatcher(new BufferSumIsEqualMatcher<uintt>(buffer, length, extra));
}

inline Matcher<int> IsEqualSum(int* buffer, size_t length,
                               const std::string& extra = "") {
  return MakeMatcher(new BufferSumIsEqualMatcher<int>(buffer, length, extra));
}

inline Matcher<floatt> IsEqualSum(floatt sum, floatt* buffer, size_t length,
                                  const std::string& extra = "") {
  return MakeMatcher(
      new BufferSumIsEqualMatcherSum<floatt>(sum, buffer, length, extra));
}

inline Matcher<uintt> IsEqualSum(uintt sum, uintt* buffer, size_t length,
                                 const std::string& extra = "") {
  return MakeMatcher(
      new BufferSumIsEqualMatcherSum<uintt>(sum, buffer, length, extra));
}

inline Matcher<int> IsEqualSum(int sum, int* buffer, size_t length,
                               const std::string& extra = "") {
  return MakeMatcher(
      new BufferSumIsEqualMatcherSum<int>(sum, buffer, length, extra));
}

MATCHER_P(MatrixValuesAreNotEqual, value, "") {
  return utils::areNotEqual(arg, value);
}

MATCHER_P(MatrixExIsEqual, buffer, "") { return utils::isEqual(arg, buffer); }

#endif /* MATRIXEQ_H */
