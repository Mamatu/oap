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

#include "MatchersUtils.hpp"
#include "oapEigen.hpp"

#include "oapHostComplexMatrixApi.hpp"
#include "oapHostMemoryApi.hpp"

#include "oapHostComplexMatrixPtr.hpp"
#include "oapFuncTests.hpp"

#include <vector>
#include <iostream>

#include "HostProcedures.hpp"

using namespace ::testing;

class OapGenericApiTests_AddConst : public testing::Test
{
 public:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(OapGenericApiTests_AddConst, Test_1)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 1}, 0.);

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 0});

  oap::HostComplexMatrixUPtr matrix1 = oap::chost::NewReMatrixWithValue (1, 1, 2.);

  std::vector<math::ComplexMatrix*> outputs = {output1};
  hp.v2_add (outputs, std::vector<math::ComplexMatrix*>({matrix1}), 1.f);

  std::vector<floatt> expected1 =
  {
    3,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re.mem, output1->re.reg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_AddConst, Test_2)
{
  oap::HostProcedures hp;
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({2, 1}, 0.);
  oap::Memory memory2 = oap::host::NewMemoryWithValues ({2, 1}, 2.);

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixFromMemory (1, 1, memory1, {0, 0});
  oap::HostComplexMatrixUPtr output2 = oap::chost::NewReMatrixFromMemory (1, 1, memory1, {1, 0});

  oap::HostComplexMatrixUPtr matrix1 = oap::chost::NewReMatrixFromMemory (1, 1, memory2, {0, 0});
  oap::HostComplexMatrixUPtr matrix2 = oap::chost::NewReMatrixFromMemory (1, 1, memory2, {1, 0});

  std::vector<math::ComplexMatrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::ComplexMatrix*>({matrix1, matrix2}), 1.f);

  std::vector<floatt> expected1 =
  {
    3
  };

  std::vector<floatt> expected2 =
  {
    3
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re.mem, output1->re.reg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output2->re.mem, output2->re.reg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory1);
  std::cout << "Memory: " << std::endl << std::to_string (memory2);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapGenericApiTests_AddConst, Test_3)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({2, 1}, 0.);

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostComplexMatrixUPtr output2 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {1, 0});

  oap::HostComplexMatrixUPtr matrix1 = oap::chost::NewReMatrixWithValue (1, 1, 2.);
  oap::HostComplexMatrixUPtr matrix2 = oap::chost::NewReMatrixWithValue (1, 1, 1.);

  std::vector<math::ComplexMatrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::ComplexMatrix*>({matrix1, matrix2}), 1.f);

  std::vector<floatt> expected1 =
  {
    3,
  };

  std::vector<floatt> expected2 =
  {
    2,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re.mem, output1->re.reg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output2->re.mem, output2->re.reg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_AddConst, Test_4)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({3, 1}, 0.);

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostComplexMatrixUPtr output2 = oap::chost::NewReMatrixFromMemory (1, 1, memory, {1, 0});

  oap::HostComplexMatrixUPtr matrix1 = oap::chost::NewReMatrixWithValue (1, 1, 2.);
  oap::HostComplexMatrixUPtr matrix2 = oap::chost::NewReMatrixWithValue (1, 1, 1.);

  std::vector<math::ComplexMatrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::ComplexMatrix*>({matrix1, matrix2}), 1.f);

  std::vector<floatt> expected1 =
  {
    3,
  };

  std::vector<floatt> expected2 =
  {
    2,
  };

  std::vector<floatt> actual1 = {oap::common::GetValue (output1->re.mem, output1->re.reg, 0, 0)};
  std::vector<floatt> actual2 = {oap::common::GetValue (output2->re.mem, output2->re.reg, 0, 0)};

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_AddConst, Test_5)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({10, 10}, 0.);

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixFromMemory (3, 3, memory, {0, 0});
  oap::HostComplexMatrixUPtr output2 = oap::chost::NewReMatrixFromMemory (3, 3, memory, {4, 0});

  oap::HostComplexMatrixUPtr matrix1 = oap::chost::NewReMatrixWithValue (3, 3, 2.);
  oap::HostComplexMatrixUPtr matrix2 = oap::chost::NewReMatrixWithValue (3, 3, 1.);

  std::vector<math::ComplexMatrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::ComplexMatrix*>({matrix1, matrix2}), 1.f);

  std::vector<floatt> expected1 =
  {
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
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
      actual1.push_back (oap::common::GetValue (output1->re.mem, output1->re.reg, x, y));
      actual2.push_back (oap::common::GetValue (output2->re.mem, output2->re.reg, x, y));
    }
  }

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapGenericApiTests_AddConst, Test_6)
{
  oap::HostProcedures hp;
  oap::Memory memory = oap::host::NewMemoryWithValues ({10, 10}, 0.);

  oap::HostComplexMatrixUPtr output1 = oap::chost::NewReMatrixFromMemory (3, 3, memory, {0, 0});
  oap::HostComplexMatrixUPtr output2 = oap::chost::NewReMatrixFromMemory (2, 2, memory, {4, 0});

  oap::HostComplexMatrixUPtr matrix1 = oap::chost::NewReMatrixWithValue (3, 3, 2.);
  oap::HostComplexMatrixUPtr matrix2 = oap::chost::NewReMatrixWithValue (2, 2, 1.);

  std::vector<math::ComplexMatrix*> outputs = {output1, output2};
  hp.v2_add (outputs, std::vector<math::ComplexMatrix*>({matrix1, matrix2}), 1.f);

  std::vector<floatt> expected1 =
  {
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
  };

  std::vector<floatt> expected2 =
  {
    2, 2,
    2, 2,
  };

  std::vector<floatt> actual1;
  std::vector<floatt> actual2;

  for (uintt x = 0; x < 3; ++x)
  {
    for (uintt y = 0; y < 3; ++y)
    {
      actual1.push_back (oap::common::GetValue (output1->re.mem, output1->re.reg, x, y));
    }
  }

  for (uintt x = 0; x < 2; ++x)
  {
    for (uintt y = 0; y < 2; ++y)
    {
      actual2.push_back (oap::common::GetValue (output2->re.mem, output2->re.reg, x, y));
    }
  }

  std::cout << "Memory: " << std::endl << std::to_string (memory);

  EXPECT_EQ (expected1, actual1);
  EXPECT_EQ (expected2, actual2);

  oap::host::DeleteMemory (memory);
}
