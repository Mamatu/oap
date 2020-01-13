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
#include <functional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "oapMemory.h"

class OapMemoryManagementTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapMemoryManagementTests, Test_1)
{
  auto newFunc = [](size_t size)
  {
    return new float[size];
  };

  auto deleteFunc = [](float* memory)
  {
    delete[] memory;
  };

  oap::MemoryManagement<float*, decltype(newFunc), decltype(deleteFunc), nullptr> memoryMng (newFunc, deleteFunc);

  float* mem = memoryMng.allocate (15);
  EXPECT_TRUE(memoryMng.deallocate (mem));
}

namespace api
{
float* newFunc (size_t size)
{
  return new float[size];
}

void deleteFunc (float* memory)
{
  delete[] memory;
}
}

TEST_F(OapMemoryManagementTests, Test_2)
{
  oap::MemoryManagement<float*, decltype(api::newFunc), decltype(api::deleteFunc), nullptr> memoryMng (api::newFunc, api::deleteFunc);

  float* mem = memoryMng.allocate (15);
  mem = memoryMng.reuse (mem);

  EXPECT_FALSE (memoryMng.deallocate (mem));
  EXPECT_TRUE (memoryMng.deallocate (mem));
}

TEST_F(OapMemoryManagementTests, Test_3)
{
  auto newFunc = [](size_t size)
  {
    return new float[size];
  };

  auto deleteFunc = [](float* memory)
  {
    delete[] memory;
  };

  oap::MemoryManagement<float*, decltype(newFunc), decltype(deleteFunc), nullptr> memoryMng (std::move(newFunc), std::move(deleteFunc));

  float* mem = memoryMng.allocate (15);
  mem = memoryMng.reuse (mem);

  EXPECT_FALSE (memoryMng.deallocate (mem));
  EXPECT_TRUE (memoryMng.deallocate (mem));
}

TEST_F(OapMemoryManagementTests, Test_4)
{
  auto newFunc = [](size_t size)
  {
    return new float[size];
  };

  auto deleteFunc = [](float* memory)
  {
    delete[] memory;
  };

  oap::MemoryManagement<float*, decltype(newFunc), decltype(deleteFunc), nullptr> memoryMng (newFunc, deleteFunc);

  float* mem = memoryMng.allocate (15);
  float* mem1 = memoryMng.allocate (25);
  mem = memoryMng.reuse (mem);

  EXPECT_TRUE (memoryMng.deallocate (mem1));
  EXPECT_FALSE (memoryMng.deallocate (mem));
  EXPECT_TRUE (memoryMng.deallocate (mem));
}

TEST_F(OapMemoryManagementTests, BuildTests)
{
  {
    auto newFunc = [](size_t size)
    {
      return new float[size];
    };

    auto deleteFunc = [](float* memory)
    {
      delete[] memory;
    };

    oap::MemoryManagement<float*, decltype(newFunc), decltype(deleteFunc), nullptr> memoryMng (newFunc, deleteFunc);
    float* mem = memoryMng.allocate (15);
    EXPECT_TRUE (memoryMng.deallocate (mem));
  }
  {
    auto newFunc = [](size_t size)
    {
      return new float[size];
    };

    auto deleteFunc = [](float* memory)
    {
      delete[] memory;
    };

    oap::MemoryManagement<float*, decltype(newFunc), decltype(deleteFunc), nullptr> memoryMng (std::move(newFunc), std::move(deleteFunc));
  }
  {
    oap::MemoryManagement<float*, decltype(api::newFunc), decltype(api::deleteFunc), nullptr> memoryMng (api::newFunc, api::deleteFunc);
  }
  {
    oap::MemoryManagement<float*, std::function<float*(size_t)>, std::function<void(float*)>, nullptr> memoryMng (api::newFunc, api::deleteFunc);
  }
}
