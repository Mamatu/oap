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

#ifndef OAP_FUNC_TESTS_H
#define OAP_FUNC_TESTS_H

#include <gtest/gtest.h>

#include "oapHostComplexMatrixApi.hpp"
#include "oapCudaMatrixUtils.hpp"

#include "oapDeviceComplexMatrixUPtr.hpp"
#include "oapHostComplexMatrixUPtr.hpp"

namespace oap
{
namespace func
{

template<typename Func, typename ExpectedCallback>
void test (const std::vector<floatt>& revalues, const std::vector<floatt>& imvalues, size_t columns, size_t rows, Func&& func, ExpectedCallback&& ecallback)
{
  debugAssert (revalues.empty () || revalues.size() == columns * rows);
  debugAssert (imvalues.empty () || imvalues.size() == columns * rows);

  size_t length = columns * rows;

  oap::DeviceComplexMatrixUPtr dmatrix = nullptr;
  oap::DeviceComplexMatrixUPtr dmatrix1 = nullptr;
  oap::HostComplexMatrixUPtr hmatrix = nullptr;

  floatt nan = std::numeric_limits<floatt>::quiet_NaN();
  std::vector<floatt> nanvalues (columns * rows, nan);

  if (!revalues.empty() && !imvalues.empty())
  {
    dmatrix = oap::cuda::NewDeviceMatrix (columns, rows);
    oap::cuda::CopyHostArrayToDeviceMatrix (dmatrix, revalues.data(), imvalues.data(), length);
    dmatrix1 = oap::cuda::NewDeviceMatrix (columns, rows);
    oap::cuda::CopyHostArrayToDeviceMatrix (dmatrix1, nanvalues.data(), nanvalues.data(), length);
    hmatrix = oap::chost::NewComplexMatrix (columns, rows);
  }
  else if (!revalues.empty())
  {
    dmatrix = oap::cuda::NewDeviceReMatrix (columns, rows);
    oap::cuda::CopyHostArrayToDeviceReMatrix (dmatrix, revalues.data(), length);
    dmatrix1 = oap::cuda::NewDeviceReMatrix (columns, rows);
    oap::cuda::CopyHostArrayToDeviceReMatrix (dmatrix1, nanvalues.data(), length);
    hmatrix = oap::chost::NewReMatrix (columns, rows);
  }
  else if (!imvalues.empty())
  {
    dmatrix = oap::cuda::NewDeviceImMatrix (columns, rows);
    oap::cuda::CopyHostArrayToDeviceImMatrix (dmatrix, imvalues.data(), length);
    dmatrix1 = oap::cuda::NewDeviceImMatrix (columns, rows);
    oap::cuda::CopyHostArrayToDeviceImMatrix (dmatrix1, nanvalues.data(), length);
    hmatrix = oap::chost::NewImMatrix (columns, rows);
  }
  
  debugAssertMsg (dmatrix != nullptr, "Probably was provided both empty revalues and imvalues");

  func (dmatrix1, dmatrix);
  oap::cuda::CopyDeviceMatrixToHostMatrix (hmatrix.get(), dmatrix1.get());

  ecallback (hmatrix.get ());
}

template<typename Func>
void test_defaultExpected (const std::vector<floatt>& revalues, const std::vector<floatt>& imvalues, size_t columns, size_t rows, Func&& func, const std::vector<floatt>& expected_revalues, const std::vector<floatt>& expected_imvalues)
{
  test (revalues, imvalues, columns, rows, func, [&expected_revalues, &expected_imvalues](const math::ComplexMatrix* hmatrix)
  {
    if (hmatrix->re.mem.ptr)
    {
      std::vector<floatt> reVs (hmatrix->re.mem.ptr, hmatrix->re.mem.ptr + (gColumns (hmatrix) * gRows (hmatrix)));
      ASSERT_EQ(expected_revalues, reVs);
    }

    if (hmatrix->im.mem.ptr)
    {
      std::vector<floatt> imVs (hmatrix->im.mem.ptr, hmatrix->im.mem.ptr + (gColumns (hmatrix) * gRows (hmatrix)));
      ASSERT_EQ(expected_imvalues, imVs);
    }
  });
}

template<typename Func>
void test_getVectors (const std::vector<floatt>& revalues, const std::vector<floatt>& imvalues, size_t columns, size_t rows, Func&& func, std::vector<floatt>& o_revalues, std::vector<floatt>& o_imvalues)
{
  test (revalues, imvalues, columns, rows, func, [&o_revalues, &o_imvalues](const math::ComplexMatrix* hmatrix)
  {
    if (hmatrix->re.mem.ptr)
    {
      std::vector<floatt> reVs (hmatrix->re.mem.ptr, hmatrix->re.mem.ptr + (gColumns (hmatrix) * gRows (hmatrix)));
      o_revalues = std::move (reVs);
    }

    if (hmatrix->im.mem.ptr)
    {
      std::vector<floatt> imVs (hmatrix->im.mem.ptr, hmatrix->im.mem.ptr + (gColumns (hmatrix) * gRows (hmatrix)));
      o_imvalues = std::move (imVs);
    }
  });
}

}}
#endif
