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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "oapHostMemoryApi.h"

class OapHostMemoryApiTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapHostMemoryApiTests, Test_1)
{
  oap::Memory* memory1 = oap::host::NewMemoryWithValues ({1, 1}, 2.12f);
  oap::Memory* memory2 = oap::host::NewMemoryWithValues ({1, 1}, 1.34f);

  oap::host::Copy (memory1, {0, 0}, memory2, {{0, 0}, {1, 1}});

  EXPECT_EQ (1.34f, oap::utils::GetValue (memory1, {{0, 0}, {1, 1}}, 0, 0));

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, Test_2)
{
  oap::Memory* memory1 = oap::host::NewMemoryWithValues ({1, 2}, 2.12f);
  oap::Memory* memory2 = oap::host::NewMemoryWithValues ({1, 1}, 1.34f);

  oap::host::Copy (memory1, {0, 0}, memory2, {{0, 0}, {1, 1}});

  EXPECT_EQ (1.34f, oap::utils::GetValue (memory1, OAP_NONE_REGION, 0, 0));
  EXPECT_EQ (2.12f, oap::utils::GetValue (memory1, OAP_NONE_REGION, 0, 1));

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, Test_3)
{
  oap::Memory* memory1 = oap::host::NewMemoryWithValues ({10, 10}, 2.12f);
  oap::Memory* memory2 = oap::host::NewMemoryWithValues ({9, 9}, 1.34f);

  oap::host::Copy (memory1, {1, 1}, memory2, {{1, 1}, {3, 3}});

  for (uintt x = 0; x < 3; ++x)
  {
    for (uintt y = 0; y < 3; ++y)
    {
      EXPECT_EQ (1.34f, oap::utils::GetValue (memory1, {{1, 1}, {3, 3}}, x, y));
    }
  }
  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapHostMemoryApiTests, Test_4)
{
  oap::Memory* memory1 = oap::host::NewMemoryWithValues ({1, 3}, 2.12f);
  oap::Memory* memory2 = oap::host::NewMemoryWithValues ({1, 2}, 1.34f);

  oap::host::Copy (memory1, {0, 1}, memory2, {{0, 0}, {1, 2}});

  EXPECT_EQ (2.12f, oap::utils::GetValue (memory1, OAP_NONE_REGION, 0, 0));
  EXPECT_EQ (1.34f, oap::utils::GetValue (memory1, OAP_NONE_REGION, 0, 1));
  EXPECT_EQ (1.34f, oap::utils::GetValue (memory1, OAP_NONE_REGION, 0, 2));

  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}
