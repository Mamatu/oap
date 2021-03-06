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

#include "DeviceImagesLoader.h"
#include "KernelExecutor.h"
#include "oapCudaMatrixUtils.h"
#include "oapDeviceComplexMatrixUPtr.h"

#include "PngFile.h"
#include "CuProceduresApi.h"

#include <memory>

using namespace ::testing;

class OapDeviceImagesLoaderTests : public testing::Test
{
 public:
  virtual void SetUp() { oap::cuda::Context::Instance().create(); }

  virtual void TearDown() { oap::cuda::Context::Instance().destroy(); }
};

TEST_F(OapDeviceImagesLoaderTests, LoadImagesAllocDeallocTest)
{
  oap::ImagesLoader::Info info("oap2dt3d/data/images_monkey_125", "image_", 125, true);
  oap::DeviceImagesLoader* ddl = oap::ImagesLoader::createImagesLoader<oap::PngFile, oap::DeviceImagesLoader>(info);
  oap::ImagesLoader* dl = oap::ImagesLoader::createImagesLoader<oap::PngFile, oap::ImagesLoader>(info);

  math::ComplexMatrix* matrix = ddl->createRowVector(0);
  math::ComplexMatrix* dmatrix = ddl->createDeviceRowVector(0);
  math::ComplexMatrix* matrix1 = dl->createRowVector(0);

  uint drows = oap::cuda::GetRows(dmatrix);
  uint dcolumns = oap::cuda::GetColumns(dmatrix);
  EXPECT_EQ(gColumns (matrix), gColumns (matrix1));
  EXPECT_EQ(gRows (matrix), gRows (matrix1));
  EXPECT_EQ(gRows (matrix), drows);
  EXPECT_EQ(gColumns (matrix), dcolumns);

  oap::host::DeleteMatrix(matrix);
  oap::host::DeleteMatrix(matrix1);
  oap::cuda::DeleteDeviceMatrix(dmatrix);
}

TEST_F(OapDeviceImagesLoaderTests, SquareMatrixAllocationTest)
{
  oap::ImagesLoader::Info info("oap2dt3d/data/images_monkey_125", "image_", 125, true);
  oap::DeviceImagesLoader* ddl = oap::ImagesLoader::createImagesLoader<oap::PngFile, oap::DeviceImagesLoader>(info);
  oap::RecToSquareApi smatrix (ddl->createMatrix(), true);

  math::MatrixInfo minfo = smatrix.getMatrixInfo ();

  oap::DeviceMatrixUPtr submatrix = smatrix.createDeviceSubMatrix (0, 1);
  oap::DeviceMatrixUPtr rowVector = smatrix.createDeviceRowVector (0);
  {
    oap::DeviceMatrixUPtr submatrix1 = smatrix.createDeviceSubMatrix (0, 100);
    oap::DeviceMatrixUPtr submatrix2 = smatrix.createDeviceSubMatrix (1, 100);
    oap::DeviceMatrixUPtr submatrix3 = smatrix.createDeviceSubMatrix (2, 100);
    oap::DeviceMatrixUPtr submatrix4 = smatrix.createDeviceSubMatrix (3, 100);
    EXPECT_THROW(smatrix.createDeviceSubMatrix (100000, 100), std::runtime_error);
  }
/*

  {
    math::ComplexMatrix* sm1 = smatrix.createDeviceSubMatrix (0, 100);
    sm1 = smatrix.getDeviceSubMatrix (1, 100, sm1);
    sm1 = smatrix.getDeviceSubMatrix (1, 100, sm1);
    EXPECT_THROW(smatrix.getDeviceSubMatrix (10000000, 100, sm1), std::runtime_error);
    oap::cuda::DeleteDeviceMatrix (sm1);
  }

  {
    oap::DeviceMatrixUPtr submatrix1 = smatrix.getDeviceSubMatrix (0, 100, nullptr);
    EXPECT_TRUE (submatrix1.get() != nullptr);
    EXPECT_EQ (100, oap::cuda::GetRows(submatrix1));
    EXPECT_EQ (oap::cuda::GetRows(submatrix1), oap::cuda::GetRows(submatrix1));
    EXPECT_EQ (oap::cuda::GetColumns(submatrix1), oap::cuda::GetColumns(submatrix1));
  }

  {
    oap::DeviceMatrixUPtr vec = smatrix.getDeviceRowVector (0, nullptr);
    EXPECT_TRUE (vec.get() != nullptr);
    EXPECT_EQ (1, oap::cuda::GetRows(vec));
    EXPECT_EQ (oap::cuda::GetRows(vec), oap::cuda::GetRows(vec));
    EXPECT_EQ (oap::cuda::GetColumns(vec), oap::cuda::GetColumns(vec));
  }

  EXPECT_EQ(1, CudaUtils::GetRows (submatrix));
  EXPECT_EQ(minfo.columns (), CudaUtils::GetColumns (submatrix));
*/
}

TEST_F(OapDeviceImagesLoaderTests, DISABLED_SquareMatrixSubMatrix)
{
  oap::ImagesLoader::Info info("oap2dt3d/data/images_monkey_125", "image_", 125, true);
  oap::DeviceImagesLoader* ddl = oap::ImagesLoader::createImagesLoader<oap::PngFile, oap::DeviceImagesLoader>(info);
  oap::RecToSquareApi smatrix (ddl->createMatrix (), true);

  math::MatrixInfo minfo = smatrix.getMatrixInfo ();

}

