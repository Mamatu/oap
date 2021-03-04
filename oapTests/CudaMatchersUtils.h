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

#ifndef OAP_CUDA_MATCHER_UTILS_H
#define OAP_CUDA_MATCHER_UTILS_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <limits>

#include "Utils.h"

#include "MatchersImpl.h"
#include "CudaMatchersImpl.h"

#include "InfoType.h"
#include "oapHostMatrixUtils.h"

namespace oap { namespace cuda {

  inline Matcher<math::ComplexMatrix*> MatrixIsEqualHK (math::ComplexMatrix* matrix, const InfoType& infoType = InfoType())
  {
    return MakeMatcher(new oap::cuda::MatrixIsEqualMatcherHK (matrix, infoType));
  }

  inline Matcher<math::ComplexMatrix*> MatrixIsEqualKH (math::ComplexMatrix* matrix, const InfoType& infoType = InfoType())
  {
    return MakeMatcher(new oap::cuda::MatrixIsEqualMatcherKH (matrix, infoType));
  }

  inline Matcher<math::ComplexMatrix*> MatrixIsEqualKK (math::ComplexMatrix* matrix, const InfoType& infoType = InfoType())
  {
    return MakeMatcher(new oap::cuda::MatrixIsEqualMatcherKK (matrix, infoType));
  }

  inline Matcher<math::ComplexMatrix*> MatrixHasValues (math::ComplexMatrix* matrix, const InfoType& infoType = InfoType())
  {
    return MakeMatcher(new oap::cuda::MatrixHasValuesMatcher(matrix, infoType));
  }

  inline Matcher<math::ComplexMatrix*> MatrixIsDiagonal(floatt value) {
    return MakeMatcher(new oap::cuda::MatrixIsDiagonalMatcher(value));
  }

  inline Matcher<math::ComplexMatrix*> MatrixIsIdentity() {
    return MakeMatcher(new oap::cuda::MatrixIsIdentityMatcher());
  }

  inline Matcher<math::ComplexMatrix*> MatrixHasValues(floatt value) {
    return MakeMatcher(new oap::cuda::MatrixValuesAreEqualMatcher(value));
  }

}}

#endif
