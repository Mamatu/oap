/*
 * Copyright 2016 - 2021 Marcin Matula
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

#ifndef MOCKUTILS_H
#define MOCKUTILS_H

#include "gtest/gtest.h"
//#include "gtest/internal/gtest-internal.hpp"
#include "gmock/gmock.h"
#include <limits>

#include "Utils.hpp"

#include "MatchersImpl.hpp"

#include "InfoType.hpp"
#include "oapHostComplexMatrixApi.hpp"

inline Matcher<math::ComplexMatrix*> MatrixIsEqual(math::ComplexMatrix* matrix, const InfoType& infoType = InfoType())
{
  return MakeMatcher(new MatrixIsEqualMatcher(matrix, infoType));
}

inline Matcher<math::ComplexMatrix*> MatrixHasValues (math::ComplexMatrix* matrix, const InfoType& infoType = InfoType())
{
  return MakeMatcher(new MatrixHasValuesMatcher(matrix, infoType));
}

inline Matcher<math::ComplexMatrix*> MatrixContainsDiagonalValues (math::ComplexMatrix* matrix)
{
  return MakeMatcher(new MatrixContainsDiagonalValuesMatcher(matrix));
}

inline Matcher<math::ComplexMatrix*> MatrixIsDiagonal(floatt value, const InfoType& infoType = InfoType()) {
  return MakeMatcher(new MatrixIsDiagonalMatcher(value, infoType));
}

inline Matcher<math::ComplexMatrix*> MatrixIsIdentity(const InfoType& infoType = InfoType()) {
  return MakeMatcher(new MatrixIsIdentityMatcher (infoType));
}

inline Matcher<math::ComplexMatrix*> MatrixHasValues(floatt value) {
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

inline Matcher<std::string> StringIsEqual(const std::string& str2, const std::string& path1, const std::string& path2) {
  return MakeMatcher(
      new StringIsEqualMatcher(str2, path1, path2));
}

inline Matcher<math::ComplexMatrix*> MatrixIsUpperTriangular (const InfoType& infoType = InfoType())
{
  return MakeMatcher(new MatrixIsUpperTriangularMatcher (infoType));
}

template <typename CalcApi>
Matcher<math::ComplexMatrix*> MatrixIsOrthogonal (CalcApi& calcApi, const InfoType& infoType = InfoType())
{
  return MakeMatcher(new MatrixIsOrthogonalMatcher<CalcApi> (calcApi, infoType));
}

MATCHER_P(MatrixValuesAreNotEqual, value, "") {
  return utils::areNotEqual(arg, value);
}

MATCHER_P(MatrixExIsEqual, buffer, "") { return utils::isEqual(arg, buffer); }

#endif /* MATRIXEQ_H */
