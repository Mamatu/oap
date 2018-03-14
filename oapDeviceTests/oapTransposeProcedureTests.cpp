#include "gtest/gtest.h"
#include "Matrix.h"
#include "Config.h"
#include "MatchersUtils.h"
#include "oapCudaMatrixUtils.h"
#include "oapHostMatrixUtils.h"
#include "KernelExecutor.h"
#include "MatrixProcedures.h"

class OapTransposeTests : public testing::Test {
 public:
  CuMatrix* m_cuMatrix;
  CUresult status;

  virtual void SetUp() {
    status = CUDA_SUCCESS;
    oap::cuda::Context::Instance().create();
    m_cuMatrix = new CuMatrix();
  }

  virtual void TearDown() {
    delete m_cuMatrix;
    oap::cuda::Context::Instance().destroy();
  }
};

TEST_F(OapTransposeTests, DeviceNoTransposeTest) {
  math::Matrix* hostMatrix = oap::host::NewReMatrix(1000, 1, 2);
  math::Matrix* hostMatrixT = oap::host::NewReMatrix(1, 1000, 0);

  math::Matrix* dMatrix = oap::cuda::NewDeviceReMatrix(1000, 1);
  math::Matrix* dMatrixT = oap::cuda::NewDeviceReMatrix(1, 1000);

  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrix, hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrixT, hostMatrixT);

  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrixT, dMatrixT);

  EXPECT_THAT(hostMatrixT, MatrixHasValues(0.f));

  oap::host::DeleteMatrix(hostMatrix);
  oap::host::DeleteMatrix(hostMatrixT);

  oap::cuda::DeleteDeviceMatrix(dMatrix);
  oap::cuda::DeleteDeviceMatrix(dMatrixT);
}

TEST_F(OapTransposeTests, DeviceTransposeTest) {
  math::Matrix* hostMatrix = oap::host::NewReMatrix(1000, 1, 2);
  math::Matrix* hostMatrixT = oap::host::NewReMatrix(1, 1000, 0);

  math::Matrix* dMatrix = oap::cuda::NewDeviceReMatrix(1000, 1);
  math::Matrix* dMatrixT = oap::cuda::NewDeviceReMatrix(1, 1000);

  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrix, hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrixT, hostMatrixT);

  m_cuMatrix->transposeMatrix(dMatrixT, dMatrix);

  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrixT, dMatrixT);

  EXPECT_THAT(hostMatrixT, MatrixHasValues(2));

  oap::host::DeleteMatrix(hostMatrix);
  oap::host::DeleteMatrix(hostMatrixT);

  oap::cuda::DeleteDeviceMatrix(dMatrix);
  oap::cuda::DeleteDeviceMatrix(dMatrixT);
}
