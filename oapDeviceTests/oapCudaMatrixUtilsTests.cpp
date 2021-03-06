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

#include "ByteBuffer.h"
#include "CuProceduresApi.h"
#include "Config.h"
#include "CudaUtils.h"

#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"

#include "oapHostComplexMatrixUPtr.h"
#include "oapDeviceComplexMatrixUPtr.h"

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

  for (size_t idx = 0; idx < columns * rows; ++idx)
  {
    *GetRePtrIndex (hmatrix, idx) = idx;
  }

  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hmatrix);

  oap::utils::ByteBuffer buffer;
  oap::cuda::SaveMatrix (dmatrix, buffer);

  oap::DeviceMatrixUPtr cmatrix = oap::cuda::LoadMatrix (buffer);

  EXPECT_TRUE(cuApi.compare (dmatrix, cmatrix));
}

TEST_F(OapCudaMatrixUtilsTests, SaveLoadMatrixToFile)
{
  std::string path = oap::utils::Config::getFileInTmp ("device_tests/OapCudaMatrixUtilsTests_SaveLoadMatrix.bin");

  oap::CuProceduresApi cuApi;

  size_t columns = 32;
  size_t rows = 16;

  oap::HostMatrixUPtr hmatrix = oap::host::NewReMatrix (columns, rows);
  oap::DeviceMatrixUPtr dmatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hmatrix);

  auto save = [&](math::ComplexMatrix* matrix)
  {
    oap::utils::ByteBuffer buffer;
    oap::cuda::SaveMatrix (matrix, buffer);
    buffer.fwrite (path);
  };

  auto load = [&]() -> oap::DeviceMatrixUPtr
  {
    oap::utils::ByteBuffer buffer (path);
    return oap::DeviceMatrixUPtr(oap::cuda::LoadMatrix (buffer));
  };

  save (dmatrix);
  auto cmatrix = load ();

  EXPECT_TRUE(cuApi.compare (dmatrix, cmatrix));
}

TEST_F(OapCudaMatrixUtilsTests, NewDeviceMatrixHostRefTest)
{
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixHostRef (hostMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_TRUE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_TRUE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixHostRef (hostMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_TRUE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_FALSE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixHostRef (hostMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_FALSE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_TRUE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
}

TEST_F(OapCudaMatrixUtilsTests, NewDeviceMatrixDeviceRefTest)
{
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixHostRef (hostMatrix);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixDeviceRef(deviceMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_TRUE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_TRUE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixHostRef (hostMatrix);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixDeviceRef(deviceMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_TRUE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_FALSE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixHostRef (hostMatrix);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixDeviceRef(deviceMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_FALSE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_TRUE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
}

TEST_F(OapCudaMatrixUtilsTests, NewDeviceMatrixCopyOfHostMatrixTest)
{
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_TRUE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_TRUE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_TRUE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_FALSE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr matrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
    EXPECT_EQ (rows, oap::cuda::GetRows (matrix));
    EXPECT_EQ (columns, oap::cuda::GetColumns (matrix));
    EXPECT_FALSE (oap::cuda::GetReValuesPtr (matrix) != nullptr);
    EXPECT_TRUE (oap::cuda::GetImValuesPtr (matrix) != nullptr);
  }
}

TEST_F(OapCudaMatrixUtilsTests, NewHostMatrixCopyOfDeviceMatrixTest)
{
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
    oap::HostMatrixUPtr matrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix(deviceMatrix);
    EXPECT_EQ (rows, gRows (matrix));
    EXPECT_EQ (columns, gColumns (matrix));
    EXPECT_TRUE (matrix->re.mem.ptr != nullptr);
    EXPECT_TRUE (matrix->im.mem.ptr != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
    oap::HostMatrixUPtr matrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix(deviceMatrix);
    EXPECT_EQ (rows, gRows (matrix));
    EXPECT_EQ (columns, gColumns (matrix));
    EXPECT_TRUE (matrix->re.mem.ptr != nullptr);
    EXPECT_FALSE (matrix->im.mem.ptr != nullptr);
  }
  {
    const uintt rows = 10;
    const uintt columns = 5;
    oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 2.f);
    oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);
    oap::HostMatrixUPtr matrix = oap::cuda::NewHostMatrixCopyOfDeviceMatrix(deviceMatrix);
    EXPECT_EQ (rows, gRows (matrix));
    EXPECT_EQ (columns, gColumns (matrix));
    EXPECT_FALSE (matrix->re.mem.ptr != nullptr);
    EXPECT_TRUE (matrix->im.mem.ptr != nullptr);
  }
}

TEST_F(OapCudaMatrixUtilsTests, SetZeroRow_1)
{
  const uintt rows = 10;
  const uintt columns = 10;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 1.f);
  oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);

  oap::cuda::SetZeroRow (deviceMatrix, 0);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hostMatrix, deviceMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      if (x == 0)
      {
        EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      }
      else
      {
        EXPECT_EQ (1.f, hostMatrix->re.mem.ptr[x + columns * y]);
      }
    }
  }
  printf ("%s\n", oap::host::to_string(hostMatrix.get()).c_str());
}

TEST_F(OapCudaMatrixUtilsTests, SetZeroRow_2)
{
  const uintt rows = 10;
  const uintt columns = 10;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 1.f);
  oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);

  oap::cuda::SetZeroRow (deviceMatrix, 1);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hostMatrix, deviceMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      if (x == 1)
      {
        EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      }
      else
      {
        EXPECT_EQ (1.f, hostMatrix->re.mem.ptr[x + columns * y]);
      }
    }
  }
  printf ("%s\n", oap::host::to_string(hostMatrix.get()).c_str());
}

TEST_F(OapCudaMatrixUtilsTests, SetZeroMatrix_1)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 1.f);
  oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);

  oap::cuda::SetZeroMatrix (deviceMatrix);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hostMatrix, deviceMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      EXPECT_EQ (0, hostMatrix->im.mem.ptr[x + columns * y]);
    }
  }
}

TEST_F(OapCudaMatrixUtilsTests, SetZeroMatrix_2)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 1.f);
  oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);

  oap::cuda::SetZeroMatrix (deviceMatrix);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hostMatrix, deviceMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      EXPECT_EQ (nullptr, hostMatrix->im.mem.ptr);
    }
  }
}

TEST_F(OapCudaMatrixUtilsTests, SetZeroMatrix_3)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 1.f);
  oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);

  oap::cuda::SetZeroMatrix (deviceMatrix);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hostMatrix, deviceMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (nullptr, hostMatrix->re.mem.ptr);
      EXPECT_EQ (0, hostMatrix->im.mem.ptr[x + columns * y]);
    }
  }
}

TEST_F(OapCudaMatrixUtilsTests, SetZeroMatrix_4)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 1.f);
  oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);

  oap::cuda::SetZeroReMatrix (deviceMatrix);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hostMatrix, deviceMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      EXPECT_EQ (nullptr, hostMatrix->im.mem.ptr);
    }
  }
}

TEST_F(OapCudaMatrixUtilsTests, SetZeroMatrix_5)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 1.f);
  oap::DeviceMatrixUPtr deviceMatrix = oap::cuda::NewDeviceMatrixCopyOfHostMatrix (hostMatrix);

  oap::cuda::SetZeroImMatrix (deviceMatrix);

  oap::cuda::CopyDeviceMatrixToHostMatrix (hostMatrix, deviceMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (nullptr, hostMatrix->re.mem.ptr);
      EXPECT_EQ (0, hostMatrix->im.mem.ptr[x + columns * y]);
    }
  }
}

TEST_F(OapCudaMatrixUtilsTests, GetDiagonal_1)
{
  const uintt rows = 4;
  const uintt columns = 4;

  oap::DeviceMatrixUPtr hostMatrix = oap::cuda::NewDeviceReMatrixWithValue (columns, rows, 1.f);
  for (uintt x = 0; x < 4; ++x)
  {
    for (uintt y = 0; y < 4; ++y)
    {
      if (x != y)
      {
        oap::cuda::SetReValue (hostMatrix, x, y, 2.);
      }
    }
  }

  EXPECT_DOUBLE_EQ(1.f, oap::cuda::GetReDiagonal (hostMatrix, 0));
  EXPECT_DOUBLE_EQ(1.f, oap::cuda::GetReDiagonal (hostMatrix, 1));
  EXPECT_DOUBLE_EQ(1.f, oap::cuda::GetReDiagonal (hostMatrix, 2));
  EXPECT_DOUBLE_EQ(1.f, oap::cuda::GetReDiagonal (hostMatrix, 3));
}

TEST_F(OapCudaMatrixUtilsTests, GetDiagonal_2)
{
  const uintt rows = 6;
  const uintt columns = 6;

  oap::DeviceMatrixUPtr hostMatrix = oap::cuda::NewDeviceReMatrixWithValue (columns, rows, 10.f);
  for (uintt x = 0; x < 6; ++x)
  {
    for (uintt y = 0; y < 6; ++y)
    {
      if (x == y)
      {
        oap::cuda::SetReValue (hostMatrix, x, y, static_cast<floatt>(x));
      }
    }
  }

  EXPECT_DOUBLE_EQ(0.f, oap::cuda::GetReDiagonal (hostMatrix, 0));
  EXPECT_DOUBLE_EQ(1.f, oap::cuda::GetReDiagonal (hostMatrix, 1));
  EXPECT_DOUBLE_EQ(2.f, oap::cuda::GetReDiagonal (hostMatrix, 2));
  EXPECT_DOUBLE_EQ(3.f, oap::cuda::GetReDiagonal (hostMatrix, 3));
  EXPECT_DOUBLE_EQ(4.f, oap::cuda::GetReDiagonal (hostMatrix, 4));
  EXPECT_DOUBLE_EQ(5.f, oap::cuda::GetReDiagonal (hostMatrix, 5));
}
