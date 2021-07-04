#include "oapHostComplexMatrixApi.hpp"
#include "MatchersUtils.hpp"
#include "HostKernel.hpp"
#include "HostProcedures.hpp"

#include "oapHostComplexMatrixPtr.hpp"

#include "oapTensorProductTests_Data_1.hpp"

class OapTensorProductTests : public testing::Test {
 public:
  OapTensorProductTests() {}

  virtual ~OapTensorProductTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapTensorProductTests, Test_1)
{
  using namespace oapTensorProduct_Data::Test_1;

  oap::HostComplexMatrixPtr hostM1 = oap::chost::NewReMatrix(1, 2);
  oap::HostComplexMatrixPtr hostM2 = oap::chost::NewReMatrix(1, 1);

  oap::HostComplexMatrixPtr ehoutput = oap::chost::NewReMatrix(1, 2);
  oap::HostComplexMatrixPtr houtput = oap::chost::NewReMatrix(1, 2);

  oap::chost::CopyArrayToReMatrix (ehoutput, t_outputValues);
  oap::chost::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::chost::CopyArrayToReMatrix (hostM2, t_reValues2);

  oap::HostProcedures hp;
  oap::generic::Dim32 dims
  {{
    {1, 2},
    {1, 2},
    {1, 1}
  }};

  hp.tensorProduct (houtput, hostM1, hostM2, dims);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

TEST_F(OapTensorProductTests, Test_2)
{
  using namespace oapTensorProduct_Data::Test_1;

  oap::HostComplexMatrixPtr hostM1 = oap::chost::NewReMatrix(2, 1);
  oap::HostComplexMatrixPtr hostM2 = oap::chost::NewReMatrix(1, 1);

  oap::HostComplexMatrixPtr ehoutput = oap::chost::NewReMatrix(2, 1);
  oap::HostComplexMatrixPtr houtput = oap::chost::NewReMatrix(2, 1);

  oap::chost::CopyArrayToReMatrix (ehoutput, t_outputValues);
  oap::chost::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::chost::CopyArrayToReMatrix (hostM2, t_reValues2);

  oap::HostProcedures hp;
  oap::generic::Dim32 dims
  {{
    {2, 1},
    {2, 1},
    {1, 1}
  }};

  hp.tensorProduct (houtput, hostM1, hostM2, dims);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

