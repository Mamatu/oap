/*
 * Copyright 2016 - 2018 Marcin Matula
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
#include "gmock/gmock.h"

#include "Config.h"
#include "CudaBuffer.h"
#include "KernelExecutor.h"

template<typename T>
class TBuffer : public utils::Buffer<T, oap::cuda::HtoDMemUtl>
{
  public:

    template<typename Arg>
    uintt convertSize() const
    {
      return utils::Buffer<T, oap::cuda::HtoDMemUtl>::template convertSize<Arg>();
    }
};

class OapCudaBufferTests : public testing::Test {
 public:
  OapCudaBufferTests() {}

  virtual ~OapCudaBufferTests() {}

  virtual void SetUp()
  {
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown()
  {
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapCudaBufferTests, SimpleBufferTest)
{
  oap::cuda::HtoDBuffer<int> buffer;
  for (int value = 0; value < 10000; ++value)
  {
    buffer.push_back (value);
  }

  for (int value = 0; value < 10000; ++value)
  {
    EXPECT_EQ(value, buffer.get(value));
  }

  for (int value = 10000 - 1; value >= 0; --value)
  {
    EXPECT_EQ(value, buffer.get(value));
  }
}

TEST_F(OapCudaBufferTests, ReallocTest)
{
  oap::cuda::HtoDBuffer<int> buffer;

  buffer.realloc (100);

  uintt index = buffer.push_back (2);
  uintt index1 = buffer.push_back (20);
  uintt index2 = buffer.push_back (25);

  EXPECT_EQ(2, buffer.get (index));
  EXPECT_EQ(20, buffer.get (index1));
  EXPECT_EQ(25, buffer.get (index2));

  buffer.realloc (200);
  EXPECT_EQ(2, buffer.get (index));
  EXPECT_EQ(20, buffer.get (index1));
  EXPECT_EQ(25, buffer.get (index2));
  EXPECT_THROW (buffer.get (index2 + 10), std::runtime_error);
}

TEST_F(OapCudaBufferTests, ConvertSizeTest)
{
  TBuffer<int> tbuffer;
  TBuffer<char> tbuffer1;
  EXPECT_EQ(sizeof(double) / sizeof(int), tbuffer.template convertSize<double>());
  EXPECT_EQ(sizeof(double) / sizeof(char), tbuffer1.template convertSize<double>());
  EXPECT_EQ(sizeof(int) / sizeof(char), tbuffer1.template convertSize<int>());
}

TEST_F(OapCudaBufferTests, ConvertBufferTest)
{
  oap::cuda::HtoDBuffer<int> buffer;
  std::vector<uintt> indices;

  for (int value = 0; value < 10000; ++value)
  {
    floatt v = value + .5;
    indices.push_back (buffer.push_back (v));
  }

  for (int value = 0; value < 10000 * 2; value = value + 2)
  {
    floatt expected = (value / 2.) + .5;
    EXPECT_EQ(expected, buffer.get<floatt>(value));
  }

  for (int value = 10000 * 2 - 2; value >= 0; value = value - 2)
  {
    floatt expected = (value / 2.) + .5;
    EXPECT_EQ(expected, buffer.get<floatt>(value));
  }

  for (uintt idx = 0; idx < indices.size(); idx++)
  {
    uintt idx1 = indices[idx];
    floatt expected = (idx) + .5;
    EXPECT_EQ(expected, buffer.get<floatt>(idx1));
  }
}

TEST_F(OapCudaBufferTests, WriteReadBufferTest)
{
  oap::cuda::HtoDBuffer<floatt> buffer;
  std::string test_path = utils::Config::getPathInTmp("device_tests");
  std::string file = test_path + "OapCudaBufferTests_WriteReadBufferTest.bin";

  for (int idx = 0; idx < 10000; ++idx)
  {
    floatt v = idx + .5;
    buffer.push_back (v);
  }

  buffer.write (file);

  oap::cuda::HtoDBuffer<floatt> buffer1;
  buffer1.read (file);

  for (int idx = 0; idx < 10000; ++idx)
  {
    floatt expected = (idx) + .5;
    EXPECT_EQ(expected, buffer.get(idx));
  }
}
