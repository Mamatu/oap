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
  oap::HostProcedures hostProcedures;

  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (1, 10, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (10, 1, 2);

  math::ComplexMatrix* houtput = oap::host::NewReMatrix(10, 10);

  hostProcedures.dotProduct (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapDotProductTests, Shared_Test_1)
{
  oap::HostProcedures hostProcedures;

  math::ComplexMatrix* hostM1 = oap::host::NewReMatrixWithValue (1, 4, 2);
  math::ComplexMatrix* hostM2 = oap::host::NewReMatrixWithValue (4, 1, 2);

  math::ComplexMatrix* houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput, MatrixHasValues(4));

  oap::host::DeleteMatrix(houtput);
  oap::host::DeleteMatrix(hostM1);
  oap::host::DeleteMatrix(hostM2);
}

TEST_F(OapDotProductTests, Shared_Test_2)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4 * 4));
}

TEST_F(OapDotProductTests, Shared_Test_3)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (1, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (4, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4));
}

TEST_F(OapDotProductTests, Shared_Test_4)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_5)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_6)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (1, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (4, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (4);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4));
}

TEST_F(OapDotProductTests, Shared_Test_7)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.setMaxThreadsPerBlock (4);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_8)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (4);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_9)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (9);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_10)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (1, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (4, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (1);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(4));
}

TEST_F(OapDotProductTests, Shared_Test_11)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 1, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (1, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(1, 1);

  hostProcedures.setMaxThreadsPerBlock (1);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_12)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (4, 4, 2);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (4, 4, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(4, 4);

  hostProcedures.setMaxThreadsPerBlock (1);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(16));
}

TEST_F(OapDotProductTests, Shared_Test_13)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixUPtr hostM1 = oap::host::NewReMatrixWithValue (1, 64, 3);
  oap::HostMatrixUPtr hostM2 = oap::host::NewReMatrixWithValue (64, 1, 2);

  oap::HostMatrixUPtr houtput = oap::host::NewReMatrix(64, 64);

  hostProcedures.setMaxThreadsPerBlock (1024);
  hostProcedures.dotProductShared (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(3 * 2));
}

TEST_F(OapDotProductTests, Test_CustomDim_1)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (4, 2, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (3, 4, 0);

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
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 2, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (4, 5, 0);

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
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (6, 1, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 6, 0);

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

  EXPECT_EQ (5, GetReIndex (houtput, 0));
}

TEST_F(OapDotProductTests, Test_CustomDim_4)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (10, 10, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (3, 10, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrixWithValue (3, 10, 1);

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
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (3, 3, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 12, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 12);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(3));
}

TEST_F(OapDotProductTests, Test_Periodic_2)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 2000, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  EXPECT_THAT(houtput.get(), MatrixHasValues(5));
}

TEST_F(OapDotProductTests, Test_Periodic_3)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 2000, 1);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    *GetRePtrIndex (hostM2, idx) = idx;
  }

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 2000);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += GetReIndex (hostM2, idx1 * 5 + i);
    }
    floatt value = GetReIndex (houtput, idx);
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput);
  }
}

TEST_F(OapDotProductTests, Test_Periodic_4)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 10, 1);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    *GetRePtrIndex (hostM2, idx) = idx;
  }

  oap::HostMatrixPtr houtput = oap::host::NewReMatrix(1, 10);

  hostProcedures.dotProductPeriodic (houtput, hostM1, hostM2);

  for (uintt idx = 0; idx < 10; ++idx)
  {
    uintt idx1 = idx / 5;
    floatt sum = 0.;
    for (uintt i = 0; i < 5; ++i)
    {
      sum += GetReIndex (hostM2, idx1 * 5 + i);
    }
    floatt value = GetReIndex (houtput, idx);
    ASSERT_DOUBLE_EQ (sum, value) << "houtput: " << oap::host::to_string(houtput) << " hostM2: " << oap::host::to_string(hostM2);
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_1)
{
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (10, 10, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (3, 1000, 1);

  oap::HostMatrixPtr houtput = oap::host::NewReMatrixWithValue (3, 1000, 1);

  oap::generic::Dim32 dims
  {{
    {2, 10},
    {10, 10},
    {2, 10}
  }};

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
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (5, 5, 1);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 2000, 1);

  *GetRePtrIndex (hostM1, 24) = 2;
  *GetRePtrIndex (hostM1, 23) = 2;
  *GetRePtrIndex (hostM1, 22) = 2;
  *GetRePtrIndex (hostM1, 21) = 2;
  *GetRePtrIndex (hostM1, 20) = 2;

  oap::HostMatrixPtr houtput = oap::host::NewReMatrixWithValue (1, 2000, 1);

  oap::generic::Dim32 dims
  {{
    {1, 4},
    {5, 4},
    {1, 5}
  }};

  hostProcedures.dotProductDimPeriodic (houtput, hostM1, hostM2, dims);

  for (uintt idx = 0; idx < 2000; ++idx)
  {
    if ((idx + 1) % 5 == 0)
    {
      ASSERT_DOUBLE_EQ (1, GetReIndex (houtput, idx)) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
    else
    {
      ASSERT_DOUBLE_EQ (5, GetReIndex (houtput, idx)) << "IDX: " << idx << " houtput: " << oap::host::to_string(houtput);
    }
  }
}

TEST_F(OapDotProductTests, Test_DimPeriodic_3)
{
  using namespace oapDotProduct_Data::Test_4;
  oap::HostProcedures hostProcedures;

  oap::HostMatrixPtr houtput = oap::host::NewReMatrixWithValue (1, 536, 0);
  oap::HostMatrixPtr hostM1 = oap::host::NewReMatrixWithValue (3, 3, 0);
  oap::HostMatrixPtr hostM2 = oap::host::NewReMatrixWithValue (1, 402, 0);

  oap::generic::Dim32 dims
  {{
    {1, 3},
    {3, 3},
    {1, 3}
  }};

  oap::host::CopyArrayToReMatrix (hostM1, t_reValues1);
  oap::host::CopyArrayToReMatrix (hostM2, t_reValues2);

  hostProcedures.dotProductDimPeriodic (houtput, hostM1, hostM2, dims, 4);
  //PRINT_MATRIX(houtput.get());
}

