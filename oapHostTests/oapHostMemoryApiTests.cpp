/*
 * CopyHostToHostright 2016 - 2021 Marcin Matula
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "oapHostMemoryApi.h"
#include "oapHostMatrixUtils.h"

class OapHostMemoryApiTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapHostMemoryApiTests, Test_1)
{
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({1, 1}, 2.12f);
  oap::Memory memory2 = oap::host::NewMemoryWithValues ({1, 1}, 1.34f);

  oap::host::CopyHostToHost (memory1, {0, 0}, memory2, {{0, 0}, {1, 1}});

  EXPECT_EQ (1.34f, oap::common::GetValueRef (memory1, {{0, 0}, {1, 1}}, 0, 0));

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, Test_2)
{
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({1, 2}, 2.12f);
  oap::Memory memory2 = oap::host::NewMemoryWithValues ({1, 1}, 1.34f);

  oap::host::CopyHostToHost (memory1, {0, 0}, memory2, {{0, 0}, {1, 1}});

  EXPECT_EQ (1.34f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 0));
  EXPECT_EQ (2.12f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 1));

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, Test_3)
{
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({10, 10}, 2.12f);
  oap::Memory memory2 = oap::host::NewMemoryWithValues ({9, 9}, 1.34f);

  oap::host::CopyHostToHost (memory1, {1, 1}, memory2, {{1, 1}, {3, 3}});

  for (uintt x = 0; x < 3; ++x)
  {
    for (uintt y = 0; y < 3; ++y)
    {
      EXPECT_EQ (1.34f, oap::common::GetValueRef (memory1, {{1, 1}, {3, 3}}, x, y));
    }
  }
  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, Test_4)
{
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({1, 3}, 2.12f);
  oap::Memory memory2 = oap::host::NewMemoryWithValues ({1, 2}, 1.34f);

  oap::host::CopyHostToHost (memory1, {0, 1}, memory2, {{0, 0}, {1, 2}});

  EXPECT_EQ (2.12f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 0));
  EXPECT_EQ (1.34f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 1));
  EXPECT_EQ (1.34f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 2));

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, CopyTest_1)
{
  oap::Memory memory1 = oap::host::NewMemoryWithValues ({1, 8}, 0);
  oap::Memory memory2 = oap::host::NewMemoryWithValues ({1, 1}, 2.f);

  oap::host::CopyHostToHost (memory1, {0, 0}, memory2, {{0, 0}, {1, 1}});

  EXPECT_EQ (2.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 0));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 1));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 2));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 3));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 4));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 5));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 6));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 7));

  oap::host::CopyHostToHost (memory1, {0, 1}, memory2, {{0, 0}, {1, 1}});

  EXPECT_EQ (2.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 0));
  EXPECT_EQ (2.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 1));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 2));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 3));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 4));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 5));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 6));
  EXPECT_EQ (0.f, oap::common::GetValue (memory1, oap::common::OAP_NONE_REGION(), 0, 7));

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, CopyTest_2)
{
  math::ComplexMatrix* matrix1 = oap::host::NewReMatrixWithValue (1, 8, 0);
  math::ComplexMatrix* matrix2 = oap::host::NewReMatrixWithValue (1, 1, 2.f);

  oap::host::SetReMatrix (matrix1, matrix2, 0, 0);

  EXPECT_EQ (2.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 0));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 1));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 2));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 3));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 4));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 5));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 6));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 7));

  oap::host::SetReMatrix (matrix1, matrix2, 0, 1);

  EXPECT_EQ (2.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 0));
  EXPECT_EQ (2.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 1));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 2));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 3));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 4));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 5));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 6));
  EXPECT_EQ (0.f, oap::common::GetValue (matrix1->re.mem, oap::common::OAP_NONE_REGION(), 0, 7));

  oap::host::DeleteMatrix (matrix1);
  oap::host::DeleteMatrix (matrix2);
}

TEST_F(OapHostMemoryApiTests, NewMemoryBulk_Test_1)
{
  std::vector<oap::Memory> memories =
  {
    oap::host::NewMemoryWithValues ({1, 1}, 1.),
    oap::host::NewMemoryWithValues ({1, 1}, 2.),
    oap::host::NewMemoryWithValues ({1, 1}, 3.)
  };

  oap::Memory bulk1 = oap::host::NewMemoryBulk (memories, oap::DataDirection::HORIZONTAL);

  EXPECT_EQ (3, bulk1.dims.width);
  EXPECT_EQ (1, bulk1.dims.height);

  EXPECT_EQ (1., bulk1.ptr[0]);
  EXPECT_EQ (2., bulk1.ptr[1]);
  EXPECT_EQ (3., bulk1.ptr[2]);

  oap::Memory bulk2 = oap::host::NewMemoryBulk (memories, oap::DataDirection::VERTICAL);

  EXPECT_EQ (1, bulk2.dims.width);
  EXPECT_EQ (3, bulk2.dims.height);

  EXPECT_EQ (1., bulk2.ptr[0]);
  EXPECT_EQ (2., bulk2.ptr[1]);
  EXPECT_EQ (3., bulk2.ptr[2]);

  oap::host::DeleteMemory (memories[0]);
  oap::host::DeleteMemory (memories[1]);
  oap::host::DeleteMemory (memories[2]);

  oap::host::DeleteMemory (bulk1);
  oap::host::DeleteMemory (bulk2);
}

TEST_F(OapHostMemoryApiTests, NewMemoryBulk_Test_2)
{
  std::vector<oap::Memory> memories =
  {
    oap::host::NewMemoryWithValues ({2, 2}, 1.),
    oap::host::NewMemoryWithValues ({2, 2}, 2.),
    oap::host::NewMemoryWithValues ({2, 2}, 3.)
  };

  oap::Memory bulk1 = oap::host::NewMemoryBulk (memories, oap::DataDirection::HORIZONTAL);

  EXPECT_EQ (6, bulk1.dims.width);
  EXPECT_EQ (2, bulk1.dims.height);

  EXPECT_EQ (1., bulk1.ptr[0]);
  EXPECT_EQ (1., bulk1.ptr[1]);
  EXPECT_EQ (2., bulk1.ptr[2]);
  EXPECT_EQ (2., bulk1.ptr[3]);
  EXPECT_EQ (3., bulk1.ptr[4]);
  EXPECT_EQ (3., bulk1.ptr[5]);

  EXPECT_EQ (1., bulk1.ptr[6]);
  EXPECT_EQ (1., bulk1.ptr[7]);
  EXPECT_EQ (2., bulk1.ptr[8]);
  EXPECT_EQ (2., bulk1.ptr[9]);
  EXPECT_EQ (3., bulk1.ptr[10]);
  EXPECT_EQ (3., bulk1.ptr[11]);

  oap::Memory bulk2 = oap::host::NewMemoryBulk (memories, oap::DataDirection::VERTICAL);

  EXPECT_EQ (2, bulk2.dims.width);
  EXPECT_EQ (6, bulk2.dims.height);

  EXPECT_EQ (1., bulk2.ptr[0]);
  EXPECT_EQ (1., bulk2.ptr[1]);
  EXPECT_EQ (1., bulk2.ptr[2]);
  EXPECT_EQ (1., bulk2.ptr[3]);

  EXPECT_EQ (2., bulk2.ptr[4]);
  EXPECT_EQ (2., bulk2.ptr[5]);
  EXPECT_EQ (2., bulk2.ptr[6]);
  EXPECT_EQ (2., bulk2.ptr[7]);

  EXPECT_EQ (3., bulk2.ptr[8]);
  EXPECT_EQ (3., bulk2.ptr[9]);
  EXPECT_EQ (3., bulk2.ptr[10]);
  EXPECT_EQ (3., bulk2.ptr[11]);

  oap::host::DeleteMemory (memories[0]);
  oap::host::DeleteMemory (memories[1]);
  oap::host::DeleteMemory (memories[2]);

  oap::host::DeleteMemory (bulk1);
  oap::host::DeleteMemory (bulk2);
}
