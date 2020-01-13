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

#include <string>
#include <cmath>

#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"
#include "HostKernelExecutor.h"
#include "HostProcedures.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMatrixPtr.h"

class OapCrossEntropyTests : public testing::Test {
 public:
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
  
  floatt getCrossEntropy(floatt y, floatt a)
  {
    return y * std::log(a) + (1. - y) * std::log (1. - a);
  }
};

TEST_F(OapCrossEntropyTests, CrossEntropyTest1)
{
  HostProcedures cuApi;

  oap::HostMatrixPtr matrix1 = oap::host::NewReMatrixWithValue (1, 1, 0.2);
  oap::HostMatrixPtr matrix2 = oap::host::NewReMatrixWithValue (1, 1, 0.2);
  oap::HostMatrixPtr output = oap::host::NewReMatrixWithValue (1, 1, 0);

  cuApi.crossEntropy (output.get(), matrix1, matrix2);

  EXPECT_NEAR (getCrossEntropy (0.2, 0.2), output->reValues[0], 0.00001);
}

TEST_F(OapCrossEntropyTests, CrossEntropyTest2)
{
  HostProcedures cuApi;

  oap::HostMatrixPtr matrix1 = oap::host::NewReMatrixWithValue (1, 10, 0);
  oap::HostMatrixPtr matrix2 = oap::host::NewReMatrixWithValue (1, 10, 0);
  oap::HostMatrixPtr output = oap::host::NewReMatrixWithValue (1, 10, 0);

  auto getValue = [](size_t idx, size_t max)
  {
    return static_cast<floatt>(idx + 1) / max;
  };

  for (size_t idx = 0; idx < 9; ++idx)
  {
    matrix1->reValues[idx] = getValue (idx, 10);
    matrix2->reValues[idx] = getValue (idx, 10);
  }

  cuApi.crossEntropy (output.get(), matrix1, matrix2);

  for (size_t idx = 0; idx < 9; ++idx)
  {
    EXPECT_NEAR (getCrossEntropy (getValue (idx, 10), getValue (idx, 10)), output->reValues[idx], 0.00001);
  }
}

