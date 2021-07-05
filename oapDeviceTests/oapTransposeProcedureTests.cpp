#include "gtest/gtest.h"
#include "Matrix.hpp"
#include "Config.hpp"
#include "MatchersUtils.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "oapHostComplexMatrixApi.hpp"
#include "KernelExecutor.hpp"
#include "CuProceduresApi.hpp"

class OapTransposeTests : public testing::Test
{
  public:
    oap::CuProceduresApi* m_cuMatrix;
    CUresult status;

    virtual void SetUp()
    {
      status = CUDA_SUCCESS;
      oap::cuda::Context::Instance().create();
      m_cuMatrix = new oap::CuProceduresApi();
    }

    virtual void TearDown()
    {
      delete m_cuMatrix;
      oap::cuda::Context::Instance().destroy();
    }
};

TEST_F(OapTransposeTests, DeviceNoTransposeTest)
{
  math::ComplexMatrix* hostMatrix = oap::chost::NewReMatrixWithValue (1000, 1, 2);
  math::ComplexMatrix* hostMatrixT = oap::chost::NewReMatrixWithValue (1, 1000, 0);

  math::ComplexMatrix* dMatrix = oap::cuda::NewDeviceReMatrix(1000, 1);
  math::ComplexMatrix* dMatrixT = oap::cuda::NewDeviceReMatrix(1, 1000);

  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrix, hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrixT, hostMatrixT);

  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrixT, dMatrixT);

  EXPECT_THAT(hostMatrixT, MatrixHasValues(0.f));

  oap::chost::DeleteMatrix(hostMatrix);
  oap::chost::DeleteMatrix(hostMatrixT);

  oap::cuda::DeleteDeviceMatrix(dMatrix);
  oap::cuda::DeleteDeviceMatrix(dMatrixT);
}

TEST_F(OapTransposeTests, DeviceTransposeTest)
{
  math::ComplexMatrix* hostMatrix = oap::chost::NewReMatrixWithValue (1000, 1, 2);
  math::ComplexMatrix* hostMatrixT = oap::chost::NewReMatrixWithValue (1, 1000, 0);

  math::ComplexMatrix* dMatrix = oap::cuda::NewDeviceReMatrix(1000, 1);
  math::ComplexMatrix* dMatrixT = oap::cuda::NewDeviceReMatrix(1, 1000);

  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrix, hostMatrix);
  oap::cuda::CopyHostMatrixToDeviceMatrix(dMatrixT, hostMatrixT);

  m_cuMatrix->transpose(dMatrixT, dMatrix);

  oap::cuda::CopyDeviceMatrixToHostMatrix(hostMatrixT, dMatrixT);

  EXPECT_THAT(hostMatrixT, MatrixHasValues(2));

  oap::chost::DeleteMatrix(hostMatrix);
  oap::chost::DeleteMatrix(hostMatrixT);

  oap::cuda::DeleteDeviceMatrix(dMatrix);
  oap::cuda::DeleteDeviceMatrix(dMatrixT);
}
