/*
 * Copyright 2016, 2017 Marcin Matula
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
#include "KernelExecutor.h"
#include "oapCudaMatrixUtils.h"

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
