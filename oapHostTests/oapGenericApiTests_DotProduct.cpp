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

#include "gtest/gtest.h"

#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapHostMatrixUtils.h"
#include "oapHostMemoryApi.h"

#include "oapHostMatrixPtr.h"
#include "oapFuncTests.h"

#include <vector>
#include <iostream>

#include "HostProcedures.h"

using namespace ::testing;

class OapGenericApiTests_DotProduct : public testing::Test
{
 public:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(OapGenericApiTests_DotProduct, Test_1)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 1}, 0.);

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});

  oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (1, 1, 2.);

  oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixWithValue (1, 1, 1.);

  std::vector<math::Matrix*> outputs = {output1};
  hp.v2_multiply (outputs, std::vector<math::Matrix*>({matrix1}), std::vector<math::Matrix*>({matrix2}));

  std::vector<floatt> expected1 =
  {
    2,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re, output1->reReg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_DotProduct, Test_2)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 2}, 0.);

  oap::HostMatrixUPtr output11 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr output12 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 1});

  oap::HostMatrixUPtr matrix11 = oap::host::NewReMatrixWithValue (1, 1, 4.);
  oap::HostMatrixUPtr matrix12 = oap::host::NewReMatrixWithValue (1, 1, 3.);

  oap::HostMatrixUPtr matrix21 = oap::host::NewReMatrixWithValue (1, 1, 4.);
  oap::HostMatrixUPtr matrix22 = oap::host::NewReMatrixWithValue (1, 1, 3.);

  std::vector<math::Matrix*> outputs = {output11, output12};
  std::vector<math::Matrix*> matrixs1 = {matrix11, matrix12};
  std::vector<math::Matrix*> matrixs2 = {matrix21, matrix22};
  hp.v2_multiply (outputs, matrixs1, matrixs2);

  std::vector<floatt> expected1 =
  {
    16,
  };

  std::vector<floatt> expected2 =
  {
    9,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output11->re, output11->reReg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output12->re, output12->reReg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_DotProduct, Test_3)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 2}, 0.);

  oap::HostMatrixUPtr output11 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr output12 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 1});

  oap::HostMatrixUPtr matrix11 = oap::host::NewReMatrixWithValue (1, 1, 4.);
  oap::HostMatrixUPtr matrix12 = oap::host::NewReMatrixWithValue (1, 1, 3.);

  oap::HostMatrixUPtr matrix21 = oap::host::NewReMatrixWithValue (1, 1, 4.);
  oap::HostMatrixUPtr matrix22 = oap::host::NewReMatrixWithValue (1, 1, 3.);

  std::vector<math::Matrix*> outputs = {output11, output12};
  std::vector<math::Matrix*> matrixs1 = {matrix11, matrix12};
  std::vector<math::Matrix*> matrixs2 = {matrix21, matrix22};
  hp.v2_multiply (outputs, matrixs1, matrixs2);

  std::vector<floatt> expected1 =
  {
    16,
  };

  std::vector<floatt> expected2 =
  {
    9,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output11->re, output11->reReg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output12->re, output12->reReg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

