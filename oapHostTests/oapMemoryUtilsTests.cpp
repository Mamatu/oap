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

#include <iostream>

#include "Matrix.h"
#include "oapMemoryUtils.h"
#include "oapHostMemoryApi.h"
#include "oapHostMatrixUPtr.h"

namespace host {
namespace qrtest1 {
extern const char* matrix;
}
}

class OapMemoryUtilsTests : public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_1)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo};
  oap::utils::createThreadsDim (infos,
      [] (uintt x, uintt y, uintt index, uintt columns, uintt rows)
      {
        EXPECT_EQ (1, columns);
        EXPECT_EQ (1, rows);
        EXPECT_EQ(0, x);
        EXPECT_EQ(0, y);
        if (x == 0 && y == 0)
        {
          EXPECT_EQ (0, index);
        }
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_2)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<uintt> values;
  oap::utils::createThreadsDim (infos,
      [&values] (uintt x, uintt y, uintt index, uintt columns, uintt rows)
      {
        EXPECT_TRUE ((columns == 2 && rows == 1) || (columns == 1 && rows == 2));
        EXPECT_TRUE(0 <= x && x < 2);
        EXPECT_TRUE(0 <= y && y < 2);
        EXPECT_EQ(0, y);
        values.push_back (index);
      });

  std::sort(values.begin(), values.end());
  EXPECT_EQ (0, values[0]);
  EXPECT_EQ (1, values[1]);
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_3)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 2);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  oap::utils::createThreadsDim (infos,
      [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (3, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_4)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 2);
  math::MatrixInfo minfo2 (true, false, 2, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  oap::utils::createThreadsDim (infos,
      [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (9, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_5)
{
  math::MatrixInfo minfo (true, false, 1, 2);
  math::MatrixInfo minfo1 (true, false, 2, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  uintt count = 0;
  oap::utils::createThreadsDim (infos,
      [&count](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        ++count;
        EXPECT_EQ (8, columns * rows);
      });
  EXPECT_EQ (8, count);
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_6)
{
  math::MatrixInfo minfo (true, false, 1, 2);
  math::MatrixInfo minfo1 (true, false, 3, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  uintt count = 0;
  oap::utils::createThreadsDim (infos,
      [&count](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        ++count;
        EXPECT_EQ (12, columns * rows);
      });
  EXPECT_EQ (11, count);
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_7)
{
  math::MatrixInfo minfo (true, false, 1, 2);
  math::MatrixInfo minfo1 (true, false, 1, 2);
  math::MatrixInfo minfo2 (true, false, 3, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  uintt count = 0;
  oap::utils::createThreadsDim (infos,
      [&count](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        ++count;
        EXPECT_EQ (15, columns * rows);
      });
  EXPECT_EQ (13, count);
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_8)
{
  math::MatrixInfo minfo (true, false, 1, 7);
  math::MatrixInfo minfo1 (true, false, 3, 2);
  math::MatrixInfo minfo2 (true, false, 1, 1);

  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  uintt dim = 0;
  oap::utils::createThreadsDim (infos,
      [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        EXPECT_EQ (20, dim);
      });

  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo, minfo2};
  oap::utils::createThreadsDim (infos1,
      [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });

  std::vector<math::MatrixInfo> infos2 = {minfo1, minfo2, minfo};
  oap::utils::createThreadsDim (infos2,
      [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });

  std::vector<math::MatrixInfo> infos3 = {minfo2, minfo, minfo1};
  oap::utils::createThreadsDim (infos3,
      [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });

  std::vector<math::MatrixInfo> infos4 = {minfo2, minfo1, minfo};
  oap::utils::createThreadsDim (infos4,
      [&dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_9)
{
  math::MatrixInfo minfo (true, false, 2, 2);
  math::MatrixInfo minfo1 (true, false, 3, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::createThreadsDim (infos,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::createThreadsDim (infos1,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_10)
{
  math::MatrixInfo minfo (true, false, 3, 2);
  math::MatrixInfo minfo1 (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::createThreadsDim (infos,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::createThreadsDim (infos1,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_11)
{
  math::MatrixInfo minfo (true, false, 3, 2);
  math::MatrixInfo minfo1 (true, false, 1, 7);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::createThreadsDim (infos,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::createThreadsDim (infos1,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_12)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 7);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo};
  uintt count = 0;
  uintt dim = 0;
  oap::utils::createThreadsDim (infos,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        dim = columns * rows;
        ++count;
      });
  oap::utils::createThreadsDim (infos1,
      [&count, &dim](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (dim, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_13)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 1);
  math::MatrixInfo minfo2 (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  oap::utils::createThreadsDim (infos,
      [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (3, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_14)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 2);
  math::MatrixInfo minfo2 (true, false, 1, 3);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1, minfo2};
  oap::utils::createThreadsDim (infos,
      [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (6, columns * rows);
      });
  std::vector<math::MatrixInfo> infos1 = {minfo1, minfo, minfo2};
  oap::utils::createThreadsDim (infos1,
      [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (6, columns * rows);
      });
  std::vector<math::MatrixInfo> infos2 = {minfo1, minfo2, minfo};
  oap::utils::createThreadsDim (infos2,
      [](uintt x, uintt y, uintt value, uintt columns, uintt rows)
      {
        EXPECT_EQ (6, columns * rows);
      });
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_15)
{
  math::MatrixInfo minfo (true, false, 1, 1);
  math::MatrixInfo minfo1 (true, false, 1, 1);
  std::vector<math::MatrixInfo> infos = {minfo, minfo1};
  auto dim = oap::utils::createThreadsDim (infos,
      [] (uintt x, uintt y, uintt index, uintt columns, uintt rows)
      {
        std::vector<uintt> indecies = {0, 0};
        EXPECT_TRUE ((columns == 2 && rows == 1) || (columns == 1 && rows == 2));
        EXPECT_TRUE(0 <= x && x < 2);
        EXPECT_TRUE(0 <= y && y < 2);
        EXPECT_EQ(0, y);
      });

  EXPECT_TRUE (dim.first == 1 || dim.first == 2);
  EXPECT_TRUE (dim.second == 1 || dim.second == 2);
  EXPECT_TRUE (dim.first * dim.second == 2);
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_16)
{
  oap::Memory memory = oap::host::NewMemoryWithValues ({2, 1}, 0.);

  oap::HostMatrixUPtr output1 = oap::host::NewReMatrixFromMemory (1, 1, memory, {0, 0});
  oap::HostMatrixUPtr output2 = oap::host::NewReMatrixFromMemory (1, 1, memory, {1, 0});

  oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (1, 1, 2.);
  oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixWithValue (1, 1, 1.);

  std::vector<math::Matrix*> outputs = {output1, output2};

  std::vector<std::vector<math::Matrix*>> matricesArgs = {{output1, matrix1}, {output2, matrix2}};
  std::vector<math::MatrixInfo> matrixInfos = {oap::host::GetMatrixInfo (output1), oap::host::GetMatrixInfo (output2)};
  std::vector<std::vector<math::Matrix>> matrixRefs =
  {
    {oap::host::GetRefHostMatrix (matricesArgs[0][0]), oap::host::GetRefHostMatrix (matricesArgs[0][1])},
    {oap::host::GetRefHostMatrix (matricesArgs[1][0]), oap::host::GetRefHostMatrix (matricesArgs[1][1])},
  };

  std::map<uintt, uintt> matrixIdxCounter;
  std::map<std::pair<uintt, uintt>, std::vector<uintt>> map;
  std::map<uintt, uintt> outputsMap; 

  auto dim = oap::utils::createThreadsDim (matrixInfos,
      [&outputsMap, &matricesArgs, &matrixIdxCounter, &matrixRefs](uintt x, uintt y, uintt index, uintt columns, uintt rows)
      {
        const uintt arglen = matricesArgs[index].size();

        uintt& matrixIdx = matrixIdxCounter[index];

        std::vector<uintt> indecies;

        for (uintt argidx = 0; argidx < arglen; ++argidx)
        {
          indecies.push_back (oap::common::GetMemIdxFromMatrixIdx (matrixRefs[index][argidx].re, matrixRefs[index][argidx].reReg, matrixIdx));
        }

        matrixIdx = matrixIdx + 1;

        for (uintt i : indecies)
        {
          outputsMap[i]++;
        }
      });

  EXPECT_EQ (2, dim.first * dim.second);
  EXPECT_EQ (3, outputsMap[0]);
  EXPECT_EQ (1, outputsMap[1]);
  EXPECT_EQ (2, outputsMap.size());
  oap::host::DeleteMemory (memory);
}

TEST_F(OapMemoryUtilsTests, CreateThreadsBlocksTest_17)
{
  std::vector<std::vector<math::Matrix*>> matricesArgs;
  std::vector<math::MatrixInfo> matrixInfos;
  std::vector<std::vector<math::Matrix>> matrixRefs;
  for (uintt idx = 0; idx < 134; ++idx)
  {
    matricesArgs.push_back ({oap::host::NewReMatrix(1,1)});
    matrixInfos.push_back (oap::host::GetMatrixInfo(matricesArgs.back().back()));
    matrixRefs.push_back({oap::host::GetRefHostMatrix (matricesArgs.back().back())});
  }

  std::map<uintt, uintt> matrixIdxCounter;
  std::map<std::pair<uintt, uintt>, std::vector<uintt>> map;
  std::map<uintt, uintt> outputsMap; 

  auto dim = oap::utils::createThreadsDim (matrixInfos,
      [&outputsMap, &matricesArgs, &matrixIdxCounter, &matrixRefs](uintt x, uintt y, uintt index, uintt columns, uintt rows)
      {
        const uintt arglen = matricesArgs[index].size();

        uintt& matrixIdx = matrixIdxCounter[index];

        std::vector<uintt> indecies;

        for (uintt argidx = 0; argidx < arglen; ++argidx)
        {
          indecies.push_back (oap::common::GetMemIdxFromMatrixIdx (matrixRefs[index][argidx].re, matrixRefs[index][argidx].reReg, matrixIdx));
        }

        matrixIdx = matrixIdx + 1;

        for (uintt i : indecies)
        {
          outputsMap[i]++;
        }
      });

  //EXPECT_EQ (3, dim.first * dim.second);

  for (const auto& vec : matricesArgs)
  {
    oap::host::deleteMatrices (vec);
  }
}
