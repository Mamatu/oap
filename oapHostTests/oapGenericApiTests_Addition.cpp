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

class OapGenericApiTests_Addition : public testing::Test
{
 public:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(OapGenericApiTests_Addition, Test_1)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 1}, 0.);

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});

  oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (1, 1, 2.);

  oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixWithValue (1, 1, 1.);
  
  std::vector<math::Matrix*> outputs = {output1};
  hp.v2_add (outputs, std::vector<math::Matrix*>({matrix1}), std::vector<math::Matrix*>({matrix2}));

  std::vector<floatt> expected1 =
  {
    3,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re, output1->reReg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_Addition, Test_2)
{
  HostProcedures hp;
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({2, 1}, 0.);
  oap::Memory memory2 = oap::host::NewMemoryWithValues ({2, 1}, 2.);
  oap::Memory memory3 = oap::host::NewMemoryWithValues ({2, 1}, 0.);
  memory3.ptr[0] = 1;
  memory3.ptr[1] = 2;

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (1, 1, memory1, {0, 0});
  oap::HostMatrixUPtr output2 = oap::host::NewReMatrixFromMemory (1, 1, memory1, {1, 0});

  oap::HostMatrixUPtr matrix11 = oap::host::NewReMatrixFromMemory (1, 1, memory2, {0, 0});
  oap::HostMatrixUPtr matrix12 = oap::host::NewReMatrixFromMemory (1, 1, memory2, {1, 0});

  oap::HostMatrixUPtr matrix21 = oap::host::NewReMatrixFromMemory (1, 1, memory3, {0, 0});
  oap::HostMatrixUPtr matrix22 = oap::host::NewReMatrixFromMemory (1, 1, memory3, {1, 0});

  std::vector<math::Matrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::Matrix*>({matrix11, matrix12}), std::vector<math::Matrix*>({matrix21, matrix22}));

  std::vector<floatt> expected1 =
  {
    3
  };

  std::vector<floatt> expected2 =
  {
    4
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re, output1->reReg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output2->re, output2->reReg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory1);
  std::cout << "Memory: " << std::endl << std::to_string (memory2);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
  oap::host::DeleteMemory (memory3);
}

TEST_F(OapGenericApiTests_Addition, Test_3)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({2, 1}, 0.);

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr output2 = oap::host::NewReMatrixFromMemory (1, 1, memory, {1, 0});

  oap::HostMatrixUPtr matrix11 = oap::host::NewReMatrixWithValue (1, 1, 2.);
  oap::HostMatrixUPtr matrix12 = oap::host::NewReMatrixWithValue (1, 1, 1.);

  oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixWithValue (1, 1, 1.);

  std::vector<math::Matrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::Matrix*>({matrix11, matrix12}), std::vector<math::Matrix*>({matrix2, matrix2}));

  std::vector<floatt> expected1 =
  {
    3,
  };

  std::vector<floatt> expected2 =
  {
    2,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re, output1->reReg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output2->re, output2->reReg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_Addition, Test_4)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({4, 1}, 1.);

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr output2 = oap::host::NewReMatrixFromMemory (1, 1, memory, {1, 0});

  oap::HostMatrixUPtr matrix11 = oap::host::NewReMatrixWithValue (1, 1, 2.);
  oap::HostMatrixUPtr matrix12 = oap::host::NewReMatrixWithValue (1, 1, 1.);

  oap::HostMatrixUPtr matrix21 = oap::host::NewReMatrixFromMemory (1, 1, memory, {2, 0});
  oap::HostMatrixUPtr matrix22 = oap::host::NewReMatrixFromMemory (1, 1, memory, {3, 0});

  std::vector<math::Matrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::Matrix*>({matrix11, matrix12}), std::vector<math::Matrix*>({matrix21, matrix22}));

  std::vector<floatt> expected1 =
  {
    3,
  };

  std::vector<floatt> expected2 =
  {
    2,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re, output1->reReg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output2->re, output2->reReg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_Addition, Test_5)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({10, 10}, 0.);

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (3, 3, memory, {0, 0});
  oap::HostMatrixUPtr output2 = oap::host::NewReMatrixFromMemory (3, 3, memory, {4, 0});

  oap::HostMatrixUPtr matrix11 = oap::host::NewReMatrixWithValue (3, 3, 2.);
  oap::HostMatrixUPtr matrix12 = oap::host::NewReMatrixWithValue (3, 3, 1.);

  oap::HostMatrixUPtr matrix21 = oap::host::NewReMatrixWithValue (3, 3, 2.);
  oap::HostMatrixUPtr matrix22 = oap::host::NewReMatrixWithValue (3, 3, 1.);

  std::vector<math::Matrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::Matrix*>({matrix11, matrix12}), std::vector<math::Matrix*>({matrix21, matrix22}));

  std::vector<floatt> expected1 =
  {
    4, 4, 4,
    4, 4, 4,
    4, 4, 4,
  };

  std::vector<floatt> expected2 =
  {
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,
  };

  std::vector<floatt> actual1;
  std::vector<floatt> actual2;

  for (uintt x = 0; x < 3; ++x)
  {
    for (uintt y = 0; y < 3; ++y)
    {
      actual1.push_back (oap::common::GetValue (output1->re, output1->reReg, x, y));
      actual2.push_back (oap::common::GetValue (output2->re, output2->reReg, x, y));
    }
  }

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_Addition, Test_6)
{
  HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({10, 10}, 0.);
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  memory1.ptr[4] = 3.;
  memory1.ptr[5] = 3.;
  memory1.ptr[14] = 3.;
  memory1.ptr[15] = 3.;

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (3, 3, memory, {0, 0});
  oap::HostMatrixUPtr output2 = oap::host::NewReMatrixFromMemory (2, 2, memory, {4, 0});

  oap::HostMatrixUPtr matrix11 = oap::host::NewReMatrixWithValue (3, 3, 2.);
  oap::HostMatrixUPtr matrix12 = oap::host::NewReMatrixWithValue (2, 2, 1.);

  oap::HostMatrixUPtr matrix21 = oap::host::NewReMatrixFromMemory (3, 3, memory1, {0, 0});
  oap::HostMatrixUPtr matrix22 = oap::host::NewReMatrixFromMemory (2, 2, memory1, {4, 0});

  std::vector<math::Matrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::Matrix*>({matrix11, matrix12}), std::vector<math::Matrix*>({matrix21, matrix22}));

  std::vector<floatt> expected1 =
  {
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
  };

  std::vector<floatt> expected2 =
  {
    4, 4,
    4, 4,
  };

  std::vector<floatt> actual1;
  std::vector<floatt> actual2;

  for (uintt x = 0; x < 3; ++x)
  {
    for (uintt y = 0; y < 3; ++y)
    {
      actual1.push_back (oap::common::GetValue (output1->re, output1->reReg, x, y));
    }
  }

  for (uintt x = 0; x < 2; ++x)
  {
    for (uintt y = 0; y < 2; ++y)
    {
      actual2.push_back (oap::common::GetValue (output2->re, output2->reReg, x, y));
    }
  }

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
  oap::host::DeleteMemory (memory1);
}
