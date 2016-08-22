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
