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

#ifndef CUDA_MATCHERSIMPL_H
#define CUDA_MATCHERSIMPL_H

#include "MatchersImpl.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "oapHostComplexMatrixUPtr.hpp"

using ::testing::PrintToString;
using ::testing::MakeMatcher;
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

namespace oap { namespace cuda {

class MatrixValuesAreEqualMatcher : public ::MatrixValuesAreEqualMatcher
{
 public:
  MatrixValuesAreEqualMatcher(floatt value) : ::MatrixValuesAreEqualMatcher(value)
  {}

  virtual bool MatchAndExplain(math::ComplexMatrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixValuesAreEqualMatcher::MatchAndExplain (hmatrix.get(), listener);
  }
};

class MatrixIsEqualMatcherKH : public ::MatrixIsEqualMatcher
{
 public:
  MatrixIsEqualMatcherKH (math::ComplexMatrix* matrix, const InfoType& infoType)
      : ::MatrixIsEqualMatcher(matrix, infoType)
  {}

  virtual bool MatchAndExplain (math::ComplexMatrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixIsEqualMatcher::MatchAndExplain (hmatrix.get (), listener);
  }
};

class MatrixIsEqualMatcherHK : public ::MatrixIsEqualMatcher
{
 public:
  MatrixIsEqualMatcherHK (math::ComplexMatrix* matrix, const InfoType& infoType)
      : ::MatrixIsEqualMatcher (oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix), infoType)
  {}

  virtual ~MatrixIsEqualMatcherHK()
  {
    oap::chost::DeleteMatrix (m_matrix);
  }

  virtual bool MatchAndExplain (math::ComplexMatrix* matrix, MatchResultListener* listener) const override
  {
    return ::MatrixIsEqualMatcher::MatchAndExplain (matrix, listener);
  }
};

class MatrixIsEqualMatcherKK : public ::MatrixIsEqualMatcher
{
 public:
  MatrixIsEqualMatcherKK (math::ComplexMatrix* matrix, const InfoType& infoType)
      : ::MatrixIsEqualMatcher (oap::cuda::NewHostMatrixCopyOfDeviceMatrix(matrix), infoType)
  {}

  virtual ~MatrixIsEqualMatcherKK()
  {
    oap::chost::DeleteMatrix (m_matrix);
  }

  virtual bool MatchAndExplain (math::ComplexMatrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixIsEqualMatcher::MatchAndExplain (hmatrix.get(), listener);
  }
};

class MatrixHasValuesMatcher : public ::MatrixHasValuesMatcher
{
 public:
  MatrixHasValuesMatcher(math::ComplexMatrix* matrix, const InfoType& infoType)
      : ::MatrixHasValuesMatcher(matrix, infoType)
  {}

  virtual bool MatchAndExplain (math::ComplexMatrix* matrix, MatchResultListener* listener) const
  {
    oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixHasValuesMatcher::MatchAndExplain (hmatrix.get (), listener);
  }
};

class MatrixIsDiagonalMatcher : public ::MatrixIsDiagonalMatcher
{
 public:
  MatrixIsDiagonalMatcher(floatt value, const InfoType& infoType = InfoType()) : ::MatrixIsDiagonalMatcher (value, infoType) {}

  virtual bool MatchAndExplain(math::ComplexMatrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostComplexMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixIsDiagonalMatcher::MatchAndExplain (hmatrix.get(), listener);
  }
};

class MatrixIsIdentityMatcher : public MatrixIsDiagonalMatcher
{
 public:
  MatrixIsIdentityMatcher() : MatrixIsDiagonalMatcher (1.f) {}
};

}}

#endif /* CUDA_MATCHERSIMPL_H */
