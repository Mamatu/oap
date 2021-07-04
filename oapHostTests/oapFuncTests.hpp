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

#ifndef OAP_HOST_FUNC_TESTS_H
#define OAP_HOST_FUNC_TESTS_H

#include <gtest/gtest.h>

#include "oapHostComplexMatrixApi.hpp"

#include "oapHostComplexMatrixUPtr.hpp"

namespace oap
{
namespace host
{
namespace func
{

template<typename Func, typename ExpectedCallback>
void test (const std::vector<floatt>& revalues, const std::vector<floatt>& imvalues, size_t columns, size_t rows, Func&& func, ExpectedCallback&& ecallback)
{
  debugAssert (revalues.empty () || revalues.size() == columns * rows);
  debugAssert (imvalues.empty () || imvalues.size() == columns * rows);

  size_t length = columns * rows;

  oap::HostComplexMatrixUPtr dmatrix = nullptr;
  oap::HostComplexMatrixUPtr dmatrix1 = nullptr;
  oap::HostComplexMatrixUPtr hmatrix = nullptr;

  if (!revalues.empty() && !imvalues.empty())
  {
    dmatrix = oap::chost::NewHostMatrix (true, true, columns, rows);
    oap::chost::CopyHostArrayToHostMatrix (dmatrix, revalues.data(), imvalues.data(), length);
    dmatrix1 = oap::chost::NewHostMatrix (true, true, columns, rows);
    hmatrix = oap::chost::NewHostMatrix (true, true, columns, rows);
  }
  else if (!revalues.empty())
  {
    dmatrix = oap::chost::NewHostMatrix (true, false, columns, rows);
    oap::chost::CopyHostArrayToHostReMatrix (dmatrix, revalues.data(), length);
    dmatrix1 = oap::chost::NewHostMatrix (true, false, columns, rows);
    hmatrix = oap::chost::NewHostMatrix (true, false, columns, rows);
  }
  else if (!imvalues.empty())
  {
    dmatrix = oap::chost::NewHostMatrix (false, true, columns, rows);
    oap::chost::CopyHostArrayToHostImMatrix (dmatrix, imvalues.data(), length);
    dmatrix1 = oap::chost::NewHostMatrix (false, true, columns, rows);
    hmatrix = oap::chost::NewHostMatrix (false, true, columns, rows);
  }
  
  debugAssertMsg (dmatrix != nullptr, "Probably was provided both empty revalues and imvalues");

  func (dmatrix1, dmatrix);
  oap::chost::CopyHostMatrixToHostMatrix (hmatrix.get(), dmatrix1.get());

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

}}}
#endif
