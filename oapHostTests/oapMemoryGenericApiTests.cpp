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

#include "oapMemory_GenericApi.h"
#include "oapHostMemoryApi.h"
#include "oapHostMatrixUtils.h"

class OapMemoryGenericApiTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapMemoryGenericApiTests, RecognizeMemoryType_1)
{
  oap::MemoryDim dstDim = {10, 10};
  oap::MemoryLoc dstLoc = {0, 0};
  oap::MemoryDim srcDim = {10, 10};
  oap::MemoryRegion srcReg = {{0, 0}, {10, 10}};

  EXPECT_TRUE (oap::generic::isLinearMemory (dstDim, dstLoc, srcDim, srcReg));
  EXPECT_FALSE (oap::generic::isBlockMemory (dstDim, dstLoc, srcDim, srcReg));
}

TEST_F(OapMemoryGenericApiTests, RecognizeMemoryType_2)
{
  oap::Memory dstMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* dstMatrix = oap::host::NewReMatrixFromMemory (10, 10, dstMem, {0, 0});

  oap::Memory srcMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* srcMatrix = oap::host::NewReMatrixFromMemory (10, 10, srcMem, {0, 0});

  EXPECT_TRUE (oap::generic::isLinearMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));
  EXPECT_FALSE (oap::generic::isBlockMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));

  oap::host::DeleteMatrix (dstMatrix);
  oap::host::DeleteMatrix (srcMatrix);
  oap::host::DeleteMemory (dstMem);
  oap::host::DeleteMemory (srcMem);
}

TEST_F(OapMemoryGenericApiTests, RecognizeMemoryType_3)
{
  oap::Memory dstMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* dstMatrix = oap::host::NewReMatrixFromMemory (10, 2, dstMem, {0, 1});

  oap::Memory srcMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* srcMatrix = oap::host::NewReMatrixFromMemory (10, 2, srcMem, {0, 2});

  EXPECT_TRUE (oap::generic::isLinearMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));
  EXPECT_FALSE (oap::generic::isBlockMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));

  oap::host::DeleteMatrix (dstMatrix);
  oap::host::DeleteMatrix (srcMatrix);
  oap::host::DeleteMemory (dstMem);
  oap::host::DeleteMemory (srcMem);
}

TEST_F(OapMemoryGenericApiTests, RecognizeMemoryType_4)
{
  oap::Memory dstMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* dstMatrix = oap::host::NewReMatrixFromMemory (2, 10, dstMem, {1, 0});

  oap::Memory srcMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* srcMatrix = oap::host::NewReMatrixFromMemory (2, 10, srcMem, {2, 0});

  EXPECT_FALSE (oap::generic::isLinearMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));
  EXPECT_TRUE (oap::generic::isBlockMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));

  oap::host::DeleteMatrix (dstMatrix);
  oap::host::DeleteMatrix (srcMatrix);
  oap::host::DeleteMemory (dstMem);
  oap::host::DeleteMemory (srcMem);
}

TEST_F(OapMemoryGenericApiTests, RecognizeMemoryType_5)
{
  oap::Memory dstMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* dstMatrix = oap::host::NewReMatrixFromMemory (5, 5, dstMem, {1, 1});

  oap::Memory srcMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* srcMatrix = oap::host::NewReMatrixFromMemory (5, 5, srcMem, {2, 2});

  EXPECT_FALSE (oap::generic::isLinearMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));
  EXPECT_TRUE (oap::generic::isBlockMemory (dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.dims, srcMatrix->re.reg));

  oap::host::DeleteMatrix (dstMatrix);
  oap::host::DeleteMatrix (srcMatrix);
  oap::host::DeleteMemory (dstMem);
  oap::host::DeleteMemory (srcMem);
}

TEST_F(OapMemoryGenericApiTests, CopyLinear_1)
{
  oap::Memory dstMem = oap::host::NewMemoryWithValues ({10, 10}, 0.);
  math::ComplexMatrix* dstMatrix = oap::host::NewReMatrixFromMemory (10, 2, dstMem, {0, 1});

  oap::Memory srcMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* srcMatrix = oap::host::NewReMatrixFromMemory (10, 2, srcMem, {0, 2});

  oap::generic::copy (dstMatrix->re.mem.ptr, dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.ptr, srcMatrix->re.mem.dims, srcMatrix->re.reg, memcpy, memcpy);

  for (uintt x = 0; x < 10; ++x)
  {
    for (uintt y = 0; y < 10; ++y)
    {
      uintt idx = x + y * 10;
      if (y >= 1 && y < 3)
      {
        EXPECT_EQ (1., dstMatrix->re.mem.ptr[x + y * 10]) << " ("<< x << ", " << y << ", " << idx << ")";
      }
      else
      {
        EXPECT_EQ (0., dstMatrix->re.mem.ptr[x + y * 10]) << " ("<< x << ", " << y << ", " << idx << ")";
      }
    }
  }

  oap::host::DeleteMatrix (dstMatrix);
  oap::host::DeleteMatrix (srcMatrix);
  oap::host::DeleteMemory (dstMem);
  oap::host::DeleteMemory (srcMem);
}

TEST_F(OapMemoryGenericApiTests, CopyBlock_1)
{
  oap::Memory dstMem = oap::host::NewMemoryWithValues ({10, 10}, 0.);
  math::ComplexMatrix* dstMatrix = oap::host::NewReMatrixFromMemory (5, 5, dstMem, {1, 1});

  oap::Memory srcMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* srcMatrix = oap::host::NewReMatrixFromMemory (5, 5, srcMem, {2, 2});

  oap::generic::copy (dstMatrix->re.mem.ptr, dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.ptr, srcMatrix->re.mem.dims, srcMatrix->re.reg, memcpy, memcpy);

  for (uintt x = 0; x < 10; ++x)
  {
    for (uintt y = 0; y < 10; ++y)
    {
      uintt idx = x + y * 10;
      if (y >= 1 && y < 6 && x >= 1 && x < 6)
      {
        EXPECT_EQ (1., dstMatrix->re.mem.ptr[x + y * 10]) << " ("<< x << ", " << y << ", " << idx << ")";
      }
      else
      {
        EXPECT_EQ (0., dstMatrix->re.mem.ptr[x + y * 10]) << " ("<< x << ", " << y << ", " << idx << ")";
      }
    }
  }

  oap::host::DeleteMatrix (dstMatrix);
  oap::host::DeleteMatrix (srcMatrix);
  oap::host::DeleteMemory (dstMem);
  oap::host::DeleteMemory (srcMem);
}

TEST_F(OapMemoryGenericApiTests, CopyBlock_2)
{
  oap::Memory dstMem = oap::host::NewMemoryWithValues ({10, 10}, 0.);
  math::ComplexMatrix* dstMatrix = oap::host::NewReMatrixFromMemory (2, 10, dstMem, {1, 0});

  oap::Memory srcMem = oap::host::NewMemoryWithValues ({10, 10}, 1.);
  math::ComplexMatrix* srcMatrix = oap::host::NewReMatrixFromMemory (2, 10, srcMem, {2, 0});

  oap::generic::copy (dstMatrix->re.mem.ptr, dstMatrix->re.mem.dims, dstMatrix->re.reg.loc, srcMatrix->re.mem.ptr, srcMatrix->re.mem.dims, srcMatrix->re.reg, memcpy, memcpy);

  for (uintt x = 0; x < 10; ++x)
  {
    for (uintt y = 0; y < 10; ++y)
    {
      uintt idx = x + y * 10;
      if (x >= 1 && x < 3)
      {
        EXPECT_EQ (1., dstMatrix->re.mem.ptr[idx]) << " ("<< x << ", " << y << ", " << idx << ")";
      }
      else
      {
        EXPECT_EQ (0., dstMatrix->re.mem.ptr[idx]) << " ("<< x << ", " << y << ", " << idx << ")";
      }
    }
  }


  oap::host::DeleteMatrix (dstMatrix);
  oap::host::DeleteMatrix (srcMatrix);
  oap::host::DeleteMemory (dstMem);
  oap::host::DeleteMemory (srcMem);
}

