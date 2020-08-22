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
#include "oapHostMemoryApi.h"

class OapMemoryApiTests : public testing::Test
{
 public:

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapMemoryApiTests, Test_1)
{
  oap::Memory memory = oap::host::NewMemory ({1, 1});
  oap::Memory memory1 = oap::host::ReuseMemory (memory);
  oap::Memory memory2 = oap::host::ReuseMemory (memory);

  oap::host::DeleteMemory (memory);
  oap::host::DeleteMemory (memory1);
  oap::host::DeleteMemory (memory2);
}

TEST_F(OapMemoryApiTests, Test_2)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({2, 1}, 2.f);
  oap::host::DeleteMemory (memory);
}
