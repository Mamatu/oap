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
#include "gmock/gmock.h"

#include "Config.hpp"
#include "HostBuffer.hpp"

template<typename T>
class TBuffer : public oap::utils::Buffer<T, oap::utils::HostMemUtl>
{
  public:

    template<typename Arg>
    uintt getArgLength() const
    {
      return oap::utils::Buffer<T, oap::utils::HostMemUtl>::template getArgLength<Arg>();
    }
};

class OapHostBufferTests : public testing::Test {
 public:
  OapHostBufferTests() {}

  virtual ~OapHostBufferTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapHostBufferTests, InitializationTest)
{
  oap::host::HostBuffer<int> buffer;
  buffer.realloc (1);
  EXPECT_EQ(1, buffer.getLength ());
  EXPECT_EQ(sizeof(int), buffer.getSizeOfBuffer ());
}

TEST_F(OapHostBufferTests, SimpleBufferTest)
{
  oap::host::HostBuffer<int> buffer;
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

  for (int idx = 0; idx < 10000; ++idx)
  {
    EXPECT_EQ(idx, buffer.read());
  }
}

TEST_F(OapHostBufferTests, BufferInBufferTest)
{
  oap::host::HostBuffer<int> buffer;
  for (int idx = 1; idx < 100; ++idx)
  {
    buffer.push_back (idx);

    std::unique_ptr<int[]> ints (new int[idx]);

    for (int idx1 = 0; idx1 < idx; ++idx1)
    {
      ints[idx1] = idx1;
    }

    buffer.push_back (ints.get (), idx);
  }

  for (int idx = 1; idx < 100; ++idx)
  {
    EXPECT_EQ(idx, buffer.read());

    std::unique_ptr<int[]> ints (new int[idx]);

    buffer.read (ints.get (), idx);

    for (int idx1 = 0; idx1 < idx; ++idx1)
    {
      EXPECT_EQ(idx1, ints[idx1]);
    }
  }
}

TEST_F(OapHostBufferTests, ReallocTest)
{
  oap::host::HostBuffer<int> buffer;

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

TEST_F(OapHostBufferTests, ConvertSizeTest)
{
  TBuffer<int> tbuffer;
  TBuffer<char> tbuffer1;
  EXPECT_EQ(sizeof(double) / sizeof(int), tbuffer.template getArgLength<double>());
  EXPECT_EQ(sizeof(double) / sizeof(char), tbuffer1.template getArgLength<double>());
  EXPECT_EQ(sizeof(int) / sizeof(char), tbuffer1.template getArgLength<int>());
}

TEST_F(OapHostBufferTests, ConvertBufferTest)
{
  oap::host::HostBuffer<int> buffer;
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

TEST_F(OapHostBufferTests, WriteReadBufferTest)
{
  oap::host::HostBuffer<floatt> buffer;
  std::string test_path = oap::utils::Config::getPathInTmp("host_tests");
  std::string file = test_path + "OapHostBufferTests_WriteReadBufferTest.bin";

  for (int idx = 0; idx < 10000; ++idx)
  {
    floatt v = idx + .5;
    buffer.push_back (v);
  }

  buffer.fwrite (file);

  oap::host::HostBuffer<floatt> buffer1;
  buffer1.fread (file);

  for (int idx = 0; idx < 10000; ++idx)
  {
    floatt expected = (idx) + .5;
    EXPECT_EQ(expected, buffer.get(idx));
  }
}
