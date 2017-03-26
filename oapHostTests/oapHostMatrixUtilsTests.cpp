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

TEST_F(OapHostMatrixUtilsTests, WriteReadMatrix) {
  uintt columns = 10;
  uintt rows = 10;

  std::string path = "/tmp/test_file";

  math::Matrix* m1 = host::NewMatrix(true, true, columns, rows, 0);

  for (int fa = 0; fa < columns * rows; ++fa) {
    m1->reValues[fa] = fa;
    m1->imValues[fa] = fa;
  }

  bool status = host::WriteMatrix(path, m1);

  EXPECT_TRUE(status);

  if (status) {
    math::Matrix* m2 = host::ReadMatrix(path);

    EXPECT_EQ(m2->columns, m1->columns);
    EXPECT_EQ(m2->columns, columns);
    EXPECT_EQ(m2->rows, m1->rows);
    EXPECT_EQ(m2->rows, rows);

    for (int fa = 0; fa < columns * rows; ++fa) {
      EXPECT_EQ(fa, m2->reValues[fa]);
      EXPECT_EQ(fa, m2->imValues[fa]);
    }

    host::DeleteMatrix(m2);
  }
  host::DeleteMatrix(m1);
}

TEST_F(OapHostMatrixUtilsTests, WriteMatrixReadVector) {
  uintt columns = 10;
  uintt rows = 10;

  std::string path = "/tmp/test_file";

  math::Matrix* matrix = host::NewMatrix(true, true, columns, rows, 0);

  for (int fa = 0; fa < columns * rows; ++fa) {
    matrix->reValues[fa] = fa;
    matrix->imValues[fa] = fa;
  }

  bool status = host::WriteMatrix(path, matrix);

  EXPECT_TRUE(status);

  if (status) {
    size_t index = 1;
    math::Matrix* vec = host::ReadRowVector(path, index);

    EXPECT_EQ(vec->columns, matrix->columns);
    EXPECT_EQ(vec->columns, columns);
    EXPECT_EQ(vec->rows, 1);

    for (int fa = 0; fa < columns; ++fa) {
      EXPECT_EQ(fa + index * columns, vec->reValues[fa]);
      EXPECT_EQ(fa + index * columns, vec->imValues[fa]);
    }

    host::DeleteMatrix(vec);
  }
  host::DeleteMatrix(matrix);
}
