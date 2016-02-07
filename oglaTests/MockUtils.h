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
#include "InfoType.h"
#include "HostMatrixModules.h"

inline Matcher<math::Matrix*> MatrixIsEqual(
    math::Matrix* matrix, const InfoType& infoType = InfoType()) {
  return MakeMatcher(new MatrixIsEqualMatcher(matrix, infoType));
}

inline Matcher<math::Matrix*> MatrixIsDiagonal(floatt value) {
  return MakeMatcher(new MatrixIsDiagonalMatcher(value));
}

inline Matcher<math::Matrix*> MatrixIsIdentity() {
  return MakeMatcher(new MatrixIsIdentityMatcher());
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
