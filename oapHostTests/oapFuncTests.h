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

#ifndef OAP_HOST_FUNC_TESTS_H
#define OAP_HOST_FUNC_TESTS_H

#include <gtest/gtest.h>

#include "oapHostMatrixUtils.h"

#include "oapHostMatrixUPtr.h"

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

  oap::HostMatrixUPtr dmatrix = nullptr;
  oap::HostMatrixUPtr dmatrix1 = nullptr;
  oap::HostMatrixUPtr hmatrix = nullptr;

  if (!revalues.empty() && !imvalues.empty())
  {
    dmatrix = oap::host::NewHostMatrix (true, true, columns, rows);
    oap::host::CopyHostArrayToHostMatrix (dmatrix, revalues.data(), imvalues.data(), length);
    dmatrix1 = oap::host::NewHostMatrix (true, true, columns, rows);
    hmatrix = oap::host::NewHostMatrix (true, true, columns, rows);
  }
  else if (!revalues.empty())
  {
    dmatrix = oap::host::NewHostMatrix (true, false, columns, rows);
    oap::host::CopyHostArrayToHostReMatrix (dmatrix, revalues.data(), length);
    dmatrix1 = oap::host::NewHostMatrix (true, false, columns, rows);
    hmatrix = oap::host::NewHostMatrix (true, false, columns, rows);
  }
  else if (!imvalues.empty())
  {
    dmatrix = oap::host::NewHostMatrix (false, true, columns, rows);
    oap::host::CopyHostArrayToHostImMatrix (dmatrix, imvalues.data(), length);
    dmatrix1 = oap::host::NewHostMatrix (false, true, columns, rows);
    hmatrix = oap::host::NewHostMatrix (false, true, columns, rows);
  }
  
  debugAssertMsg (dmatrix != nullptr, "Probably was provided both empty revalues and imvalues");

  func (dmatrix1, dmatrix);
  oap::host::CopyHostMatrixToHostMatrix (hmatrix.get(), dmatrix1.get());

  ecallback (hmatrix.get ());
}

template<typename Func>
void test_defaultExpected (const std::vector<floatt>& revalues, const std::vector<floatt>& imvalues, size_t columns, size_t rows, Func&& func, const std::vector<floatt>& expected_revalues, const std::vector<floatt>& expected_imvalues)
{
  test (revalues, imvalues, columns, rows, func, [&expected_revalues, &expected_imvalues](const math::Matrix* hmatrix)
  {
    if (hmatrix->reValues)
    {
      std::vector<floatt> reVs (hmatrix->reValues, hmatrix->reValues + (hmatrix->columns * hmatrix->rows));
      ASSERT_EQ(expected_revalues, reVs);
    }

    if (hmatrix->imValues)
    {
      std::vector<floatt> imVs (hmatrix->imValues, hmatrix->imValues + (hmatrix->columns * hmatrix->rows));
      ASSERT_EQ(expected_imvalues, imVs);
    }
  });
}

template<typename Func>
void test_getVectors (const std::vector<floatt>& revalues, const std::vector<floatt>& imvalues, size_t columns, size_t rows, Func&& func, std::vector<floatt>& o_revalues, std::vector<floatt>& o_imvalues)
{
  test (revalues, imvalues, columns, rows, func, [&o_revalues, &o_imvalues](const math::Matrix* hmatrix)
  {
    if (hmatrix->reValues)
    {
      std::vector<floatt> reVs (hmatrix->reValues, hmatrix->reValues + (hmatrix->columns * hmatrix->rows));
      o_revalues = std::move (reVs);
    }

    if (hmatrix->imValues)
    {
      std::vector<floatt> imVs (hmatrix->imValues, hmatrix->imValues + (hmatrix->columns * hmatrix->rows));
      o_imvalues = std::move (imVs);
    }
  });
}

}}}
#endif
