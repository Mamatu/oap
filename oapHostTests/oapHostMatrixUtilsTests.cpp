#include "HostMatrixUtils.h"
#include "MatchersUtils.h"

class OapHostMatrixUtilsTests : public testing::Test {
 public:
  OapHostMatrixUtilsTests() {}

  virtual ~OapHostMatrixUtilsTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(OapHostMatrixUtilsTests, Copy) {
  const uintt columns = 11;
  const uintt rows = 15;
  math::Matrix* m1 = host::NewReMatrix(columns, rows, 1);
  math::Matrix* m2 = host::NewReMatrix(columns, rows, 0);

  host::CopyMatrix(m2, m1);

  EXPECT_THAT(m1, MatrixIsEqual(m2));

  host::DeleteMatrix(m1);
  host::DeleteMatrix(m2);
}

TEST_F(OapHostMatrixUtilsTests, SubCopy) {
  const uintt columns = 11;
  const uintt rows = 15;

  math::Matrix* m1 = host::NewReMatrix(15, 15, 1);
  math::Matrix* m2 = host::NewReMatrix(5, 6, 0);

  host::CopyMatrix(m2, m1);

  EXPECT_THAT(m2, MatrixHasValues(1));

  host::DeleteMatrix(m1);
  host::DeleteMatrix(m2);
}
