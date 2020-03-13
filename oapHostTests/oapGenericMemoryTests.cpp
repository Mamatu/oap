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

#include "oapMemory_GenericApi.h"


class OapGenericMemoryApiTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapGenericMemoryApiTests, Test_1)
{
  floatt* mem = reinterpret_cast<floatt*>(0u);
  floatt* ptrs[2];
  oap::MemoryDims dims = {5, 5};
  oap::MemoryRegion region;
  region.loc = {1, 1};
  region.dims = {2, 2};

  oap::utils::getPtrs (ptrs, mem, dims, region);

  EXPECT_EQ (reinterpret_cast<floatt*>(6u * sizeof (floatt*)), ptrs[0]);
  EXPECT_EQ (reinterpret_cast<floatt*>(11u * sizeof (floatt*)), ptrs[1]);
}

TEST_F(OapGenericMemoryApiTests, Test_2)
{
  floatt* mem = reinterpret_cast<floatt*>(0u);
  std::vector<floatt*> ptrs;
  oap::MemoryDims dims = {5, 5};
  oap::MemoryRegion region;
  region.loc = {1, 1};
  region.dims = {2, 2};

  oap::utils::getPtrs (ptrs, mem, dims, region);

  EXPECT_EQ (2, ptrs.size());
  EXPECT_EQ (reinterpret_cast<floatt*>(6u * sizeof (floatt*)), ptrs[0]);
  EXPECT_EQ (reinterpret_cast<floatt*>(11u * sizeof (floatt*)), ptrs[1]);
}

TEST_F(OapGenericMemoryApiTests, Test_3)
{
  floatt* mem = reinterpret_cast<floatt*>(0u);
  std::vector<floatt*> ptrs;
  oap::MemoryDims dims = {1, 3};
  oap::MemoryRegion region;
  region.loc = {0, 0};
  region.dims = {1, 3};

  oap::utils::getPtrs (ptrs, mem, dims, region);

  EXPECT_EQ (3, ptrs.size());
  EXPECT_EQ (reinterpret_cast<floatt*>(0u * sizeof (floatt*)), ptrs[0]);
  EXPECT_EQ (reinterpret_cast<floatt*>(1u * sizeof (floatt*)), ptrs[1]);
  EXPECT_EQ (reinterpret_cast<floatt*>(2u * sizeof (floatt*)), ptrs[2]);
}
