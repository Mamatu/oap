#include "oapHostMatrixUtils.h"
#include "MatchersUtils.h"
#include "HostKernel.h"
#include "HostProcedures.h"

#include "oapHostMatrixPtr.h"

#include "oapTensorProductTests_Data_1.h"

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

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(1, 2);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 1);

  oap::HostMatrixPtr ehoutput = oap::host::NewReMatrix(1, 2);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2);

  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);
  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);

  HostProcedures hp;
  uintt dims[3][2] =
  {
    {1, 2},
    {1, 2},
    {1, 1}
  };

  hp.tensorProduct (houtput, hostM1, hostM2, dims);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

TEST_F(OapTensorProductTests, Test_2)
{
  using namespace oapTensorProduct_Data::Test_1;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(2, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 1);

  oap::HostMatrixPtr ehoutput = oap::host::NewReMatrix(2, 1);
  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(2, 1);

  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);
  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);

  HostProcedures hp;
  uintt dims[3][2] =
  {
    {2, 1},
    {2, 1},
    {1, 1}
  };

  hp.tensorProduct (houtput, hostM1, hostM2, dims);

  EXPECT_THAT(ehoutput.get(), MatrixIsEqual(houtput.get()));
}

