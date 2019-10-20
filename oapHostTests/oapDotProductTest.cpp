#include "oapHostMatrixUtils.h"
#include "MatchersUtils.h"
#include "HostKernel.h"

#include "oapHostMatrixPtr.h"
#include "oapHostMatrixUPtr.h"

#include "oapDotProductTests_Data_1.h"
#include "oapDotProductTests_Data_2.h"
#include "oapDotProductTests_Data_3.h"
#include "oapDotProductTests_Data_4.h"

class OapDotProductTests : public testing::Test {
 public:
  OapDotProductTests() {}

  virtual ~OapDotProductTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};


TEST_F(OapDotProductTests, Test1)
{
  HostProcedures hostProcedures;

  math::Matrix* hostM1 = oap::host::NewReMatrix(1, 10, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(10, 1, 2);

  math::Matrix* houtput = oap::host::NewReMatrix(10, 10);

  hostProcedures.dotProduct (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapDotProductTests, Shared_Test_1)
{
  HostProcedures hostProcedures;

  math::Matrix* hostM1 = oap::host::NewReMatrix(1, 4, 2);
  math::Matrix* hostM2 = oap::host::NewReMatrix(4, 1, 2);

  math::Matrix* houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapDotProductTests, Shared_Test_2)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4 * 4));
}

TEST_F(OapDotProductTests, Shared_Test_3)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(1, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(4, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4));
}

TEST_F(OapDotProductTests, Shared_Test_4)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_5)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_6)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(1, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(4, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (4);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4));
}

TEST_F(OapDotProductTests, Shared_Test_7)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.setMaxThreadsPerBlock (4);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_8)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (4);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_9)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_10)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(1, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(4, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (1);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4));
}

TEST_F(OapDotProductTests, Shared_Test_11)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.setMaxThreadsPerBlock (1);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_12)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (1);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_13)
{
  HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrix(1, 64, 3);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrix(64, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(64, 64);

  hostProcedures.setMaxThreadsPerBlock (1024);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(3 * 2));
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

TEST_F(OapDotProductTests, Test_CustomDim_2)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 2, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(4, 5, 0);

  using namespace oapDotProduct_Data::Test_2;

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

TEST_F(OapDotProductTests, Test_CustomDim_3)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(6, 1, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 6, 0);

  using namespace oapDotProduct_Data::Test_3;

  oap::HostMatrixPtr ehoutput = oap::host::NewReMatrix(1, 2);

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);
  oap::host::CopyArrayToReMatrix (ehoutput, t_outputValues);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 2);

  uintt oDim[2] = {1, 1};
  uintt p1Dim[2] = {5, 1};
  uintt p2Dim[2] = {1, 5};
  hostProcedures.dotProduct (houtput, hostM1, hostM2, oDim, p1Dim, p2Dim);

  EXPECT_EQ (5, houtput->reValues[0]);
}

TEST_F(OapDotProductTests, Test_CustomDim_4)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(10, 10, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(3, 10, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 10, 1);

  uintt oDim[2] = {2, 10};
  uintt p1Dim[2] = {10, 10};
  uintt p2Dim[2] = {2, 10};
  hostProcedures.dotProduct (houtput, hostM1, hostM2, oDim, p1Dim, p2Dim);

  for (size_t idx = 0; idx < 10; ++idx)
  {
    EXPECT_DOUBLE_EQ(10, GetRe (houtput, 0, idx));
    EXPECT_DOUBLE_EQ(10, GetRe (houtput, 1, idx));
    EXPECT_DOUBLE_EQ(1, GetRe (houtput, 2, idx));
  }
}

TEST_F(OapDotProductTests, Test_Periodic_1)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(3, 3, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 12, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 12);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(3));
}

TEST_F(OapDotProductTests, Test_Periodic_2)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 2000, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(5));
}

TEST_F(OapDotProductTests, Test_Periodic_3)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 2000, 1);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    hostM2->reValues[idx] = idx;
  }

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += hostM2->reValues[idx1 * 5 + i];
    }
    floatt value = houtput->reValues[idx];
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput);
  }
}

TEST_F(OapDotProductTests, Test_Periodic_4)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 10, 1);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    hostM2->reValues[idx] = idx;
  }

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 10);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += hostM2->reValues[idx1 * 5 + i];
    }
    floatt value = houtput->reValues[idx];
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput) << " hostM2: " << oap::host::to_string(hostM2);
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_1)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(10, 10, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(3, 1000, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(3, 1000, 1);

  uintt dims[3][2] =
  {
    {2, 10},
    {10, 10},
    {2, 10}
  };

  hostProcedures.dotProductDimPeriodic (houtput, hostM1, hostM2, dims);

  for (size_t idx = 0; idx < 1000; ++idx)
  {
    ASSERT_DOUBLE_EQ(10, GetRe (houtput, 0, idx)) << "houtput: " << oap::host::to_string(houtput);
    ASSERT_DOUBLE_EQ(10, GetRe (houtput, 1, idx)) << "houtput: " << oap::host::to_string(houtput);
    ASSERT_DOUBLE_EQ(1, GetRe (houtput, 2, idx)) << "houtput: " << oap::host::to_string(houtput);
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_2)
{
  HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 2000, 1);

  hostM1->reValues[24] = 2;
  hostM1->reValues[23] = 2;
  hostM1->reValues[22] = 2;
  hostM1->reValues[21] = 2;
  hostM1->reValues[20] = 2;

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000, 1);

  uintt dims[3][2] =
  {
    {1, 4},
    {5, 4},
    {1, 5}
  };

  hostProcedures.dotProductDimPeriodic (houtput, hostM1, hostM2, dims);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    if ((idx + 1) % 5 == 0)
    {
      ASSERT_DOUBLE_EQ (1, houtput->reValues[idx]) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
    else
    {
      ASSERT_DOUBLE_EQ (5, houtput->reValues[idx]) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_3)
{
  using namespace oapDotProduct_Data::Test_4;
  HostProcedures hostProcedures;

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 536, 0);
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrix(3, 3, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrix(1, 402, 0);

  uintt dims[3][2] =
  {
    {1, 3},
    {3, 3},
    {1, 3}
  };

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);

  hostProcedures.dotProductDimPeriodic (houtput, hostM1, hostM2, dims, 4);
  //PRINT_MATRIX(houtput.get());
}

