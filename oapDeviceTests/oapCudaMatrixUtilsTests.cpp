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

#include "ByteBuffer.h"
#include "CuProceduresApi.h"
#include "Config.h"

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"

#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixUPtr.h"

using namespace ::testing;

class OapCudaMatrixUtilsTests : public testing::Test
{
 public:
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
  }

  virtual void TearDown() {
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapCudaMatrixUtilsTests, SaveLoadMatrixToBuffer)
{
  oap::CuProceduresApi cuApi;

  size_t columns = 32;
  size_t rows = 16;

  oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrix (columns, rows);
  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrixCopy (hmatrix);

  utils::ByteBuffer buffer;
  oap::cuda::SaveMatrix (dmatrix, buffer);

  oap::DeviceMatrixUPtr cmatrix = oap::cuda::LoadMatrix (buffer);

  EXPECT_TRUE(cuApi.compare (dmatrix, cmatrix));
}

TEST_F(OapCudaMatrixUtilsTests, SaveLoadMatrixToFile)
{
  std::string path = utils::Config::getFileInTmp ("device_tests/OapCudaMatrixUtilsTests_SaveLoadMatrix.bin");

  oap::CuProceduresApi cuApi;

  size_t columns = 32;
  size_t rows = 16;

  oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrix (columns, rows);
  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrixCopy (hmatrix);

  auto save = [&](math::Matrix* matrix)
  {
    utils::ByteBuffer buffer;
    oap::cuda::SaveMatrix (matrix, buffer);
    buffer.fwrite (path);
  };

  auto load = [&]() -> oap::DeviceMatrixUPtr
  {
    utils::ByteBuffer buffer (path);
    return oap::DeviceMatrixUPtr(oap::cuda::LoadMatrix (buffer));
  };

  save (dmatrix);
  auto cmatrix = load ();

  EXPECT_TRUE(cuApi.compare (dmatrix, cmatrix));
}

