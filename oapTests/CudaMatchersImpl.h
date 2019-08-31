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

#ifndef CUDA_MATCHERSIMPL_H
#define CUDA_MATCHERSIMPL_H

#include "MatchersImpl.h"
#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUPtr.h"

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

  virtual bool MatchAndExplain(math::Matrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixValuesAreEqualMatcher::MatchAndExplain (hmatrix.get(), listener);
  }
};

class MatrixIsEqualMatcherKH : public ::MatrixIsEqualMatcher
{
 public:
  MatrixIsEqualMatcherKH (math::Matrix* matrix, const InfoType& infoType)
      : ::MatrixIsEqualMatcher(matrix, infoType)
  {}

  virtual bool MatchAndExplain (math::Matrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixIsEqualMatcher::MatchAndExplain (hmatrix.get (), listener);
  }
};

class MatrixIsEqualMatcherHK : public ::MatrixIsEqualMatcher
{
 public:
  MatrixIsEqualMatcherHK (math::Matrix* matrix, const InfoType& infoType)
      : ::MatrixIsEqualMatcher (oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix), infoType)
  {}

  virtual ~MatrixIsEqualMatcherHK()
  {
    oap::host::DeleteMatrix (m_matrix);
  }

  virtual bool MatchAndExplain (math::Matrix* matrix, MatchResultListener* listener) const override
  {
    return ::MatrixIsEqualMatcher::MatchAndExplain (matrix, listener);
  }
};

class MatrixIsEqualMatcherKK : public ::MatrixIsEqualMatcher
{
 public:
  MatrixIsEqualMatcherKK (math::Matrix* matrix, const InfoType& infoType)
      : ::MatrixIsEqualMatcher (oap::cuda::NewHostMatrixCopyOfDeviceMatrix(matrix), infoType)
  {}

  virtual ~MatrixIsEqualMatcherKK()
  {
    oap::host::DeleteMatrix (m_matrix);
  }

  virtual bool MatchAndExplain (math::Matrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixIsEqualMatcher::MatchAndExplain (hmatrix.get(), listener);
  }
};

class MatrixHasValuesMatcher : public ::MatrixHasValuesMatcher
{
 public:
  MatrixHasValuesMatcher(math::Matrix* matrix, const InfoType& infoType)
      : ::MatrixHasValuesMatcher(matrix, infoType)
  {}

  virtual bool MatchAndExplain (math::Matrix* matrix, MatchResultListener* listener) const
  {
    oap::HostMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
    return ::MatrixHasValuesMatcher::MatchAndExplain (hmatrix.get (), listener);
  }
};

class MatrixIsDiagonalMatcher : public ::MatrixIsDiagonalMatcher
{
 public:
  MatrixIsDiagonalMatcher(floatt value) : ::MatrixIsDiagonalMatcher (value) {}

  virtual bool MatchAndExplain(math::Matrix* matrix, MatchResultListener* listener) const override
  {
    oap::HostMatrixUPtr hmatrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix (matrix);
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
