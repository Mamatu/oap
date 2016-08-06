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
