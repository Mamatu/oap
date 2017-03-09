#include "gtest/gtest.h"
#include "Matrix.h"
#include "Config.h"
#include "MatchersUtils.h"
#include "DeviceMatrixModules.h"
#include "HostMatrixUtils.h"
#include "KernelExecutor.h"
#include "MatrixProcedures.h"

class OapTransposeTests : public testing::Test {
 public:
  CuMatrix* m_cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    device::Context::Instance().create();
    m_cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    device::Context::Instance().destroy();
    delete m_cuMatrix;
  }
};

TEST_F(OapTransposeTests, DeviceNoTransposeTest) {
  math::Matrix* hostMatrix = host::NewReMatrix(1000, 1, 2);
  math::Matrix* hostMatrixT = host::NewReMatrix(1, 1000, 0);

  math::Matrix* dMatrix = device::NewDeviceReMatrix(1000, 1);
  math::Matrix* dMatrixT = device::NewDeviceReMatrix(1, 1000);

  device::CopyHostMatrixToDeviceMatrix(dMatrix, hostMatrix);
  device::CopyHostMatrixToDeviceMatrix(dMatrixT, hostMatrixT);

  device::CopyDeviceMatrixToHostMatrix(hostMatrixT, dMatrixT);

  EXPECT_THAT(hostMatrixT, MatrixValuesAreEqual(0));

  host::DeleteMatrix(hostMatrix);
  host::DeleteMatrix(hostMatrixT);

  device::DeleteDeviceMatrix(dMatrix);
  device::DeleteDeviceMatrix(dMatrixT);
}

TEST_F(OapTransposeTests, DeviceTransposeTest) {
  math::Matrix* hostMatrix = host::NewReMatrix(1000, 1, 2);
  math::Matrix* hostMatrixT = host::NewReMatrix(1, 1000, 0);

  math::Matrix* dMatrix = device::NewDeviceReMatrix(1000, 1);
  math::Matrix* dMatrixT = device::NewDeviceReMatrix(1, 1000);

  device::CopyHostMatrixToDeviceMatrix(dMatrix, hostMatrix);
  device::CopyHostMatrixToDeviceMatrix(dMatrixT, hostMatrixT);

  m_cuMatrix->transposeMatrix(dMatrixT, dMatrix);

  device::CopyDeviceMatrixToHostMatrix(hostMatrixT, dMatrixT);

  EXPECT_THAT(hostMatrixT, MatrixValuesAreEqual(2));

  host::DeleteMatrix(hostMatrix);
  host::DeleteMatrix(hostMatrixT);

  device::DeleteDeviceMatrix(dMatrix);
  device::DeleteDeviceMatrix(dMatrixT);
}
