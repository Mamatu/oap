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
#include <stdio.h>
#include <math.h>
#include "gtest/gtest.h"
#include "oapHostMatrixUtils.h"
#include "oapHostMatrixUPtr.h"
#include "oapHostMemoryApi.h"
#include "oapThreadsMapperApi.h"

class OapThreadsMapperApiTests : public testing::Test
{
 public:

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapThreadsMapperApiTests, GetThreadsMapperTest_1)
{
  oap::HostMatrixUPtr m1 = oap::host::NewMatrix (1, 1);
  std::vector<math::Matrix*> matrices = {m1};
  auto mapper = oap::threads::createThreadsMapper(matrices, oap::host::GetMatrixInfo, memcpy);

  ASSERT_EQ (1, mapper.getLength());

  std::unique_ptr<uintt[]> buffer (new uintt[1]);
  mapper.map(buffer.get());

  EXPECT_EQ (0, buffer[0]);
}

TEST_F(OapThreadsMapperApiTests, GetThreadsMapperTest_2)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({3, 2}, 0.);
  oap::HostMatrixUPtr m1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr m2 = oap::host::NewReMatrixFromMemory (2, 2, memory, {1, 0});
  std::vector<math::Matrix*> matrices = {m1, m2};
  auto mapper = oap::threads::createThreadsMapper(matrices, oap::host::GetMatrixInfo, memcpy);

  ASSERT_EQ (6, mapper.getLength());

  std::unique_ptr<uintt[]> buffer (new uintt[6]);
  mapper.map(buffer.get());

  EXPECT_EQ (0, buffer[0]);
  EXPECT_TRUE (MAX_UINTT == buffer[1] || 1 == buffer[1]);
  EXPECT_EQ (1, buffer[2]);
  EXPECT_TRUE (MAX_UINTT == buffer[3] || 1 == buffer[3]);
  EXPECT_EQ (1, buffer[4]);
  EXPECT_EQ (1, buffer[5]);
  EXPECT_NE (buffer[1], buffer[3]);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapThreadsMapperApiTests, GetThreadsMapperTest_3)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({1, 6}, 0.);

  oap::HostMatrixUPtr m1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr m2 = oap::host::NewReMatrixFromMemory (1, 2, memory, {0, 1});
  oap::HostMatrixUPtr m3 = oap::host::NewReMatrixFromMemory (1, 3, memory, {0, 3});

  std::vector<math::Matrix*> matrices = {m1, m2, m3};
  auto mapper = oap::threads::createThreadsMapper(matrices, oap::host::GetMatrixInfo, memcpy);

  ASSERT_EQ (6, mapper.getLength());

  std::unique_ptr<uintt[]> buffer (new uintt[6]);
  mapper.map(buffer.get());

  EXPECT_EQ (0, buffer[0]);
  EXPECT_EQ (1, buffer[1]);
  EXPECT_EQ (1, buffer[2]);
  EXPECT_EQ (2, buffer[3]);
  EXPECT_EQ (2, buffer[4]);
  EXPECT_EQ (2, buffer[5]);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapThreadsMapperApiTests, GetThreadsMapperTest_4)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({6, 1}, 0.);

  oap::HostMatrixUPtr m1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr m2 = oap::host::NewReMatrixFromMemory (2, 1, memory, {1, 0});
  oap::HostMatrixUPtr m3 = oap::host::NewReMatrixFromMemory (3, 1, memory, {3, 0});
  std::vector<math::Matrix*> matrices = {m1, m2, m3};
  auto mapper = oap::threads::createThreadsMapper(matrices, oap::host::GetMatrixInfo, memcpy);

  ASSERT_EQ (6, mapper.getLength());

  std::unique_ptr<uintt[]> buffer (new uintt[6]);
  mapper.map(buffer.get());

  EXPECT_EQ (0, buffer[0]);
  EXPECT_EQ (1, buffer[1]);
  EXPECT_EQ (1, buffer[2]);
  EXPECT_EQ (2, buffer[3]);
  EXPECT_EQ (2, buffer[4]);
  EXPECT_EQ (2, buffer[5]);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapThreadsMapperApiTests, GetThreadsMapperTest_5)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({6, 1}, 0.);

  oap::HostMatrixUPtr m1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr m2 = oap::host::NewReMatrixFromMemory (2, 1, memory, {1, 0});
  oap::HostMatrixUPtr m3 = oap::host::NewReMatrixFromMemory (3, 2, memory, {3, 0});
  std::vector<math::Matrix*> matrices = {m1, m2, m3};
  auto mapper = oap::threads::createThreadsMapper(matrices, oap::host::GetMatrixInfo, memcpy);

  ASSERT_EQ (12, mapper.getLength());

  std::vector<uintt> buffer(12, MAX_UINTT);
  mapper.map(buffer.data());

  std::vector<uintt> array =
  {
    0, 1, 1, 2, 2, 2, MAX_UINTT, MAX_UINTT, MAX_UINTT, 2, 2, 2
  };

  EXPECT_EQ (array, buffer);

  oap::host::DeleteMemory (memory);
}

TEST_F(OapThreadsMapperApiTests, GetThreadsMapperTest_6)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({5, 2}, 0.);
  oap::HostMatrixUPtr m1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr m2 = oap::host::NewReMatrixFromMemory (1, 2, memory, {1, 0});
  oap::HostMatrixUPtr m3 = oap::host::NewReMatrixFromMemory (3, 2, memory, {2, 0});
  std::vector<math::Matrix*> matrices = {m1, m2, m3};
  auto mapper = oap::threads::createThreadsMapper(matrices, oap::host::GetMatrixInfo, memcpy);

  ASSERT_EQ (10, mapper.getLength());

  std::vector<uintt> buffer(10, MAX_UINTT);
  mapper.map(buffer.data());

  std::vector<uintt> array =
  {
    0, 1, 2, 2, 2, MAX_UINTT, 1, 2, 2, 2
  };

  EXPECT_EQ (array, buffer);
  oap::host::DeleteMemory (memory);
}

TEST_F(OapThreadsMapperApiTests, GetThreadsMapperTest_7)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({2, 1}, 0.);
  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr output2 = oap::host::NewReMatrixFromMemory (1, 1, memory, {1, 0});

  std::vector<math::Matrix*> matrices = {output1, output2};

  auto mapper = oap::threads::createThreadsMapper(matrices, oap::host::GetMatrixInfo, memcpy);

  EXPECT_EQ (2, mapper.getLength());
  EXPECT_EQ (2, mapper.getWidth());
  EXPECT_EQ (1, mapper.getHeight());

  std::vector<uintt> buffer(2, MAX_UINTT);
  mapper.map(buffer.data());

  std::vector<uintt> array =
  {
    0, 1
  };

  EXPECT_EQ (array, buffer);
  oap::host::DeleteMemory (memory);
}

#if 0
TEST_F(OapThreadsMapperApiTests, GetThreadsTest_1)
{
  std::vector<oap::MemoryRegion> regions;
  regions.push_back ({{0, 0}, {1, 1}});
  regions.push_back ({{1, 1}, {1, 1}});
  regions.push_back ({{2, 2}, {1, 1}});
  regions.push_back ({{3, 3}, {1, 1}});
  regions.push_back ({{4, 4}, {1, 1}});
  EXPECT_EQ (5, oap::threads::getXThreads (regions));
  EXPECT_EQ (1, oap::threads::getYThreads (regions));
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_2)
{
  std::vector<oap::MemoryRegion> regions;
  regions.push_back ({{4, 4}, {1, 1}});
  regions.push_back ({{3, 3}, {1, 1}});
  regions.push_back ({{2, 2}, {1, 1}});
  regions.push_back ({{1, 1}, {1, 1}});
  regions.push_back ({{0, 0}, {1, 1}});
  EXPECT_EQ (5, oap::threads::getXThreads (regions));
  EXPECT_EQ (1, oap::threads::getYThreads (regions));
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_3)
{
  std::vector<oap::MemoryRegion> regions;
  regions.push_back ({{4, 4}, {1, 1}});
  regions.push_back ({{3, 3}, {1, 1}});
  regions.push_back ({{2, 2}, {1, 1}});
  regions.push_back ({{1, 1}, {2, 2}});
  regions.push_back ({{0, 0}, {2, 2}});
  EXPECT_EQ (5, oap::threads::getXThreads (regions));
  EXPECT_EQ (1, oap::threads::getYThreads (regions));
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_4)
{
  std::vector<oap::MemoryRegion> regions;
  regions.push_back ({{4, 4}, {1, 1}});
  regions.push_back ({{2, 2}, {1, 1}});
  regions.push_back ({{1, 1}, {2, 2}});
  regions.push_back ({{0, 0}, {2, 2}});
  EXPECT_EQ (4, oap::threads::getXThreads (regions));
  EXPECT_EQ (4, oap::threads::getYThreads (regions));
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_5)
{
  std::vector<oap::MemoryRegion> regions;
  regions.push_back ({{4, 4}, {1, 1}});
  regions.push_back ({{2, 2}, {1, 1}});
  regions.push_back ({{1, 1}, {2, 2}});
  regions.push_back ({{0, 0}, {6, 6}});
  EXPECT_EQ (6, oap::threads::getXThreads (regions));
  EXPECT_EQ (6, oap::threads::getYThreads (regions));
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_6)
{
  std::vector<oap::MemoryRegion> regions;
  regions.push_back ({{3, 3}, {1, 1}});
  regions.push_back ({{0, 0}, {3, 3}});
  EXPECT_EQ (4, oap::threads::getXThreads (regions));
  EXPECT_EQ (4, oap::threads::getYThreads (regions));
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_7)
{
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{3, 0}, {1, 1}});
    regions.push_back ({{0, 0}, {3, 1}});
    EXPECT_EQ (4, oap::threads::getXThreads (regions));
    EXPECT_EQ (1, oap::threads::getYThreads (regions));
  }
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{0, 3}, {1, 1}});
    regions.push_back ({{0, 0}, {1, 3}});
    EXPECT_EQ (1, oap::threads::getXThreads (regions));
    EXPECT_EQ (4, oap::threads::getYThreads (regions));
  }
}

TEST_F(OapThreadsMapperApiTests, GetThreadsTest_8)
{
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{5, 0}, {1, 1}});
    regions.push_back ({{0, 0}, {3, 1}});
    EXPECT_EQ (4, oap::threads::getXThreads (regions));
    EXPECT_EQ (1, oap::threads::getYThreads (regions));
  }
  {
    std::vector<oap::MemoryRegion> regions;
    regions.push_back ({{0, 5}, {1, 1}});
    regions.push_back ({{0, 0}, {1, 3}});
    EXPECT_EQ (1, oap::threads::getXThreads (regions));
    EXPECT_EQ (4, oap::threads::getYThreads (regions));
  }
}
#endif
