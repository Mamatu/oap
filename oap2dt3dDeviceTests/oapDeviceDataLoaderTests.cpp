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

#include "DeviceDataLoader.h"
#include "SquareMatrix.h"
#include "KernelExecutor.h"
#include "oapCudaMatrixUtils.h"
#include "oapDeviceMatrixUPtr.h"

#include "PngFile.h"

#include <memory>

using namespace ::testing;

class OapDeviceDataLoaderTests : public testing::Test
{
 public:
  virtual void SetUp() { oap::cuda::Context::Instance().create(); }

  virtual void TearDown() { oap::cuda::Context::Instance().destroy(); }
};

TEST_F(OapDeviceDataLoaderTests, LoadImagesAllocDeallocTest)
{
  oap::DataLoader::Info info("oap2dt3d/data/images_monkey_125", "image_", 125, true);
  oap::DeviceDataLoader* ddl = oap::DataLoader::createDataLoader<oap::PngFile, oap::DeviceDataLoader>(info);
  oap::DataLoader* dl = oap::DataLoader::createDataLoader<oap::PngFile, oap::DataLoader>(info);

  math::Matrix* matrix = ddl->createRowVector(0);
  math::Matrix* dmatrix = ddl->createDeviceRowVector(0);
  math::Matrix* matrix1 = dl->createRowVector(0);

  uint drows = oap::cuda::GetRows(dmatrix);
  uint dcolumns = oap::cuda::GetColumns(dmatrix);
  EXPECT_EQ(matrix->columns, matrix1->columns);
  EXPECT_EQ(matrix->rows, matrix1->rows);
  EXPECT_EQ(matrix->rows, drows);
  EXPECT_EQ(matrix->columns, dcolumns);

  oap::host::DeleteMatrix(matrix);
  oap::host::DeleteMatrix(matrix1);
  oap::cuda::DeleteDeviceMatrix(dmatrix);
}

TEST_F(OapDeviceDataLoaderTests, SquareMatrixAllocationTest)
{
  oap::DataLoader::Info info("oap2dt3d/data/images_monkey_125", "image_", 125, true);
  oap::DeviceDataLoader* ddl = oap::DataLoader::createDataLoader<oap::PngFile, oap::DeviceDataLoader>(info);
  oap::SquareMatrix smatrix (ddl);

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

  {
    math::Matrix* sm1 = smatrix.createDeviceSubMatrix (0, 100);
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
  EXPECT_EQ(minfo.m_matrixDim.columns, CudaUtils::GetColumns (submatrix));
}

TEST_F(OapDeviceDataLoaderTests, SquareMatrixSubMatrix)
{
  oap::DataLoader::Info info("oap2dt3d/data/images_monkey_125", "image_", 125, true);
  oap::DeviceDataLoader* ddl = oap::DataLoader::createDataLoader<oap::PngFile, oap::DeviceDataLoader>(info);
  oap::SquareMatrix smatrix (ddl);

  math::MatrixInfo minfo = smatrix.getMatrixInfo ();

}

