/*
 * Copyright 2016 Marcin Matula
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

#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include "gtest/gtest.h"

class OapDeviceMatrixModuleTests : public testing::Test {
 public:
  virtual void SetUp() { device::Context::Instance().create(); }

  virtual void TearDown() { device::Context::Instance().destroy(); }

  void setSubMatrixTest(uintt columns, uintt rows, float value,
                        uintt subcolumns, uint subrows, floatt subvalue,
                        uintt column, uintt row) {
    math::Matrix* hmatrix = host::NewMatrix(true, true, columns, rows, value);
    math::Matrix* dmatrix = device::NewDeviceMatrixCopy(hmatrix);

    math::Matrix* hsubmatrix =
        host::NewMatrix(true, true, subcolumns, subrows, subvalue);
    math::Matrix* dsubmatrix = device::NewDeviceMatrixCopy(hsubmatrix);

    device::SetMatrix(dmatrix, dsubmatrix, column, row);
    device::CopyDeviceMatrixToHostMatrix(hmatrix, dmatrix);

    for (uintt fa = 0; fa < subcolumns; ++fa) {
      for (uintt fb = 0; fb < subrows; ++fb) {
        EXPECT_EQ(subvalue,
                  hmatrix->reValues[(fa + column) + columns * (row + fb)]);
        EXPECT_EQ(subvalue,
                  hmatrix->imValues[(fa + column) + columns * (row + fb)]);
      }
    }

    /*for (uintt fa = subrows * subcolumns; fa < columns * rows; ++fa) {
      EXPECT_EQ(value, hmatrix->reValues[fa]);
      EXPECT_EQ(value, hmatrix->imValues[fa]);
    }*/

    device::DeleteDeviceMatrix(dmatrix);
    device::DeleteDeviceMatrix(dsubmatrix);

    host::DeleteMatrix(hmatrix);
    host::DeleteMatrix(hsubmatrix);
  }
};

TEST_F(OapDeviceMatrixModuleTests, GetColumnsTest) {
  uintt columns = 15;
  uintt rows = 10;
  math::Matrix* matrix = device::NewDeviceMatrix(true, true, columns, rows);
  uintt expected = CudaUtils::GetColumns(matrix);
  uintt tested = device::GetColumns(matrix);
  EXPECT_EQ(expected, tested);
  EXPECT_EQ(columns, tested);
  device::DeleteDeviceMatrix(matrix);
}

TEST_F(OapDeviceMatrixModuleTests, GetRowsTest) {
  uintt columns = 15;
  uintt rows = 10;
  math::Matrix* matrix = device::NewDeviceMatrix(true, true, columns, rows);
  uintt expected = CudaUtils::GetRows(matrix);
  uintt tested = device::GetRows(matrix);
  EXPECT_EQ(expected, tested);
  EXPECT_EQ(rows, tested);
  device::DeleteDeviceMatrix(matrix);
}

TEST_F(OapDeviceMatrixModuleTests, SetSubMatrix00) {
  setSubMatrixTest(10, 10, 2.f, 4, 4, 1.5f, 0, 0);
}

TEST_F(OapDeviceMatrixModuleTests, SetSubValue00) {
  setSubMatrixTest(10, 10, 2.f, 1, 1, 1.5f, 0, 0);
}
