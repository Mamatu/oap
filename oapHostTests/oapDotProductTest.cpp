#include "HostMatrixUtils.h"
#include "MatchersUtils.h"
#include "HostKernel.h"
#include "CuMatrixProcedures/CuDotProductProcedures.h"

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
  math::Matrix* hostM1 = host::NewReMatrix(1, 10, 2);
  math::Matrix* hostM2 = host::NewReMatrix(10, 1, 2);

  math::Matrix* houtput = host::NewReMatrix(10, 10);

  DotProductKernel dotPrdocutKernel(houtput, hostM1, hostM2);

  dotPrdocutKernel.executeKernelAsync();

  EXPECT_THAT(houtput, MatrixHasValues(4));

  host::DeleteMatrix(houtput);
  host::DeleteMatrix(hostM1);
  host::DeleteMatrix(hostM2);
}
