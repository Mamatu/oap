#include "oapHostMatrixUtils.h"
#include "MatchersUtils.h"
#include "HostKernel.h"
#include "CuProcedures/CuDotProductProcedures.h"

#include "oapHostMatrixPtr.h"
#include "oapDotProductTests_Data_1.h"

class OapDotProductTests : public testing::Test {
 public:
  OapDotProductTests() {}

  virtual ~OapDotProductTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};

class DotProductKernel : public HostKernel {
 public:
  DotProductKernel(math::Matrix* dst, math::Matrix* p1, math::Matrix* p2) {
    setMatrices(dst, p1, p2);
  }

  void setMatrices(math::Matrix* dst, math::Matrix* p1, math::Matrix* p2) {
    m_dst = dst;
    m_p1 = p1;
    m_p2 = p2;

    setDims(dim3(1, 1), dim3(m_dst->columns, m_dst->rows));
  }

  math::Matrix* m_dst;
  math::Matrix* m_p1;
  math::Matrix* m_p2;
  virtual void execute(const dim3& threadIdx, const dim3& blockIdx) {
    CUDA_dotProductRe(m_dst, m_p1, m_p2);
  }
};

TEST_F(OapDotProductTests, Test1) {
  math::Matrix* hostM1 = oap::host::NewReMatrix(1, 10, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(10, 1, 2);

  math::Matrix* houtput = oap::host::NewReMatrix(10, 10);

  DotProductKernel dotPrdocutKernel(houtput, hostM1, hostM2);

  dotPrdocutKernel.executeKernelAsync();

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}


TEST_F(OapDotProductTests, Test_CustomDim_1)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(4, 2, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(3, 4, 0);

  using namespace oapDotProduct_Data::Test_1;

  oap::HostMatrixPtr ehoutput = oap::host::NewReMatrix(3, 2);

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);
  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 2);

  uintt oDim[2] = {3, 2};
  uintt p1Dim[2] = {4, 2};
  uintt p2Dim[2] = {3, 4};
  hostProcedures.dotProduct (houtput, hostM1, hostM2, oDim, p1Dim, p2Dim);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}
