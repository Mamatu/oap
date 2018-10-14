#include "oapHostMatrixUtils.h"
#include "MatchersUtils.h"

#include "oapHostMatrixUPtr.h"
#include "oapHostMatrixPtr.h"
#include <functional>

class OapHostMatrixUtilsTests : public testing::Test {
 public:
  OapHostMatrixUtilsTests() {}

  virtual ~OapHostMatrixUtilsTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}

  using GetValue = std::function<floatt(uintt, uintt, uintt)>;

  static const std::string testfilepath;

  static math::Matrix* createMatrix(uintt columns, uintt rows, GetValue getValue) {
    math::Matrix* m1 = oap::host::NewMatrix(true, true, columns, rows, 0);

    for (int idx1 = 0; idx1 < columns; ++idx1) {
      for (int idx2 = 0; idx2 < rows; ++idx2) {
        uintt idx3 = idx1 + idx2 * columns;
        m1->reValues[idx3] = getValue(idx1, idx2, idx3);
        m1->imValues[idx3] = getValue(idx1, idx2, idx3);
      }
    }

    return m1;
  }
};

const std::string OapHostMatrixUtilsTests::testfilepath = "/tmp/Oap/host_tests/test_file";

TEST_F(OapHostMatrixUtilsTests, Copy) {
  const uintt columns = 11;
  const uintt rows = 15;
  math::Matrix* m1 = oap::host::NewReMatrix(columns, rows, 1);
  math::Matrix* m2 = oap::host::NewReMatrix(columns, rows, 0);

  oap::host::CopyMatrix(m2, m1);

  EXPECT_THAT(m1, MatrixIsEqual(m2));

  oap::host::DeleteMatrix(m1);
  oap::host::DeleteMatrix(m2);
}

TEST_F(OapHostMatrixUtilsTests, SubCopy) {
  const uintt columns = 11;
  const uintt rows = 15;

  math::Matrix* m1 = oap::host::NewReMatrix(15, 15, 1);
  math::Matrix* m2 = oap::host::NewReMatrix(5, 6, 0);

  oap::host::CopyMatrix(m2, m1);

  EXPECT_THAT(m2, MatrixHasValues(1));

  oap::host::DeleteMatrix(m1);
  oap::host::DeleteMatrix(m2);
}

TEST_F(OapHostMatrixUtilsTests, WriteReadMatrix) {

  math::Matrix* m1 = createMatrix(10, 10, [](uintt, uintt, uintt idx) { return idx; });

  bool status = oap::host::WriteMatrix(testfilepath, m1);

  EXPECT_TRUE(status);

  if (status) {
    math::Matrix* m2 = oap::host::ReadMatrix(testfilepath);

    EXPECT_EQ(m2->columns, m1->columns);
    EXPECT_EQ(m2->rows, m1->rows);

    for (int fa = 0; fa < m1->columns * m1->rows; ++fa) {
      EXPECT_EQ(m1->reValues[fa], m2->reValues[fa]);
      EXPECT_EQ(m1->imValues[fa], m2->imValues[fa]);
    }

    oap::host::DeleteMatrix(m2);
  }
  oap::host::DeleteMatrix(m1);
}

TEST_F(OapHostMatrixUtilsTests, WriteReadMatrixEx) {

  math::Matrix* m1 = createMatrix(10, 10, [](uintt idx, uintt, uintt) { return idx; });

  bool status = oap::host::WriteMatrix(testfilepath, m1);

  EXPECT_TRUE(status);

  if (status) {
    MatrixEx mex;
    mex.beginRow = 0;
    mex.rowsLength = 10;
 
    mex.beginColumn = 4;
    mex.columnsLength = 6;

    math::Matrix* m2 = oap::host::ReadMatrix(testfilepath, mex);

    EXPECT_EQ(m2->columns, mex.columnsLength);
    EXPECT_EQ(m2->rows, mex.rowsLength);

    for (int fa = 0; fa < m2->columns; ++fa) {
      for (int fb = 0; fb < m2->rows; ++fb) {
        int idx = fa + m2->columns * fb;
        int idx1 = (mex.beginColumn + fa) + m1->columns * (mex.beginRow + fb);
        EXPECT_EQ(m2->reValues[idx], m1->reValues[idx1]);
        EXPECT_EQ(m2->imValues[idx], m1->imValues[idx1]);
      }
    }

    oap::host::DeleteMatrix(m2);
  }
  oap::host::DeleteMatrix(m1);
}

TEST_F(OapHostMatrixUtilsTests, WriteMatrixReadVector) {

  math::Matrix* matrix = createMatrix(10, 10, [](uintt, uintt, uintt idx) { return idx; });

  bool status = oap::host::WriteMatrix(testfilepath, matrix);

  EXPECT_TRUE(status);

  if (status) {
    size_t index = 1;
    math::Matrix* vec = oap::host::ReadRowVector(testfilepath, index);

    EXPECT_EQ(vec->columns, matrix->columns);
    EXPECT_EQ(vec->rows, 1);

    for (int fa = 0; fa < matrix->columns; ++fa) {
      EXPECT_EQ(fa + index * matrix->columns, vec->reValues[fa]);
      EXPECT_EQ(fa + index * matrix->columns, vec->imValues[fa]);
    }

    oap::host::DeleteMatrix(vec);
  }
  oap::host::DeleteMatrix(matrix);
}

TEST_F(OapHostMatrixUtilsTests, CreateSubMatrixTests)
{
  auto initMatrix = [](math::Matrix* matrix)
  {
    for (uintt idx = 0; idx < matrix->columns; ++idx)
    {
      for (uintt idx1 = 0; idx1 < matrix->rows; ++idx1)
      {
        oap::host::SetReValue (matrix, idx, idx1, idx + 10 * idx1);
        oap::host::SetImValue (matrix, idx, idx1, idx + 10 * idx1);
      }
    }
  };

  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostMatrixUPtr submatrix = oap::host::CreateSubMatrix (matrix, 1, 1, 2, 2);
    EXPECT_EQ(11, submatrix->reValues[0]);
    EXPECT_EQ(12, submatrix->reValues[1]);
    EXPECT_EQ(21, submatrix->reValues[2]);
    EXPECT_EQ(22, submatrix->reValues[3]);
    EXPECT_EQ(2, submatrix->columns);
    EXPECT_EQ(2, submatrix->rows);
  }

  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostMatrixUPtr submatrix = oap::host::CreateSubMatrix (matrix, 1, 1, 6, 6);
    EXPECT_EQ(11, submatrix->reValues[0]);
    EXPECT_EQ(12, submatrix->reValues[1]);
    EXPECT_EQ(13, submatrix->reValues[2]);
    EXPECT_EQ(21, submatrix->reValues[3]);
    EXPECT_EQ(22, submatrix->reValues[4]);
    EXPECT_EQ(23, submatrix->reValues[5]);
    EXPECT_EQ(3, submatrix->columns);
    EXPECT_EQ(3, submatrix->rows);
  }

  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostMatrixUPtr submatrix = oap::host::CreateSubMatrix (matrix, 0, 0, 4, 4);

    EXPECT_EQ(0, oap::host::GetReValue(submatrix, 0, 0));
    EXPECT_EQ(1, oap::host::GetReValue(submatrix, 1, 0));
    EXPECT_EQ(2, oap::host::GetReValue(submatrix, 2, 0));
    EXPECT_EQ(3, oap::host::GetReValue(submatrix, 3, 0));

    EXPECT_EQ(10, oap::host::GetReValue(submatrix, 0, 1));
    EXPECT_EQ(11, oap::host::GetReValue(submatrix, 1, 1));
    EXPECT_EQ(12, oap::host::GetReValue(submatrix, 2, 1));
    EXPECT_EQ(13, oap::host::GetReValue(submatrix, 3, 1));

    EXPECT_EQ(20,oap::host::GetReValue(submatrix, 0, 2));
    EXPECT_EQ(21,oap::host::GetReValue(submatrix, 1, 2));
    EXPECT_EQ(22,oap::host::GetReValue(submatrix, 2, 2));
    EXPECT_EQ(23,oap::host::GetReValue(submatrix, 3, 2));

    EXPECT_EQ(30,oap::host::GetReValue(submatrix, 0, 3));
    EXPECT_EQ(31,oap::host::GetReValue(submatrix, 1, 3));
    EXPECT_EQ(32,oap::host::GetReValue(submatrix, 2, 3));
    EXPECT_EQ(33,oap::host::GetReValue(submatrix, 3, 3));
  }
}

TEST_F(OapHostMatrixUtilsTests, GetSubMatrixTest)
{
  auto initMatrix = [](math::Matrix* matrix)
  {
    for (uintt idx = 0; idx < matrix->columns; ++idx)
    {
      for (uintt idx1 = 0; idx1 < matrix->rows; ++idx1)
      {
        oap::host::SetReValue (matrix, idx, idx1, idx + 10 * idx1);
        oap::host::SetImValue (matrix, idx, idx1, idx + 10 * idx1);
      }
    }
  };

  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    math::Matrix* submatrix = oap::host::CreateSubMatrix (matrix, 1, 1, 2, 2);

    EXPECT_EQ(2, submatrix->columns);
    EXPECT_EQ(2, submatrix->rows);

    EXPECT_EQ(11, submatrix->reValues[0]);
    EXPECT_EQ(12, submatrix->reValues[1]);
    EXPECT_EQ(21, submatrix->reValues[2]);
    EXPECT_EQ(22, submatrix->reValues[3]);

    math::Matrix* smatrix = oap::host::GetSubMatrix (matrix, 0, 0, submatrix);
    EXPECT_EQ(smatrix, submatrix);

    EXPECT_EQ(2, smatrix->columns);
    EXPECT_EQ(2, smatrix->rows);

    EXPECT_EQ(0, smatrix->reValues[0]);
    EXPECT_EQ(1, smatrix->reValues[1]);
    EXPECT_EQ(10, smatrix->reValues[2]);
    EXPECT_EQ(11, smatrix->reValues[3]);

    smatrix = oap::host::GetSubMatrix (matrix, 1, 1, smatrix);
    EXPECT_EQ(smatrix, submatrix);

    EXPECT_EQ(2, smatrix->columns);
    EXPECT_EQ(2, smatrix->rows);

    EXPECT_EQ(11, smatrix->reValues[0]);
    EXPECT_EQ(12, smatrix->reValues[1]);
    EXPECT_EQ(21, smatrix->reValues[2]);
    EXPECT_EQ(22, smatrix->reValues[3]);

    smatrix = oap::host::GetSubMatrix (matrix, 2, 2, smatrix);
    EXPECT_EQ(smatrix, submatrix);
    EXPECT_EQ(22, smatrix->reValues[0]);
    EXPECT_EQ(23, smatrix->reValues[1]);
    EXPECT_EQ(32, smatrix->reValues[2]);
    EXPECT_EQ(33, smatrix->reValues[3]);
    EXPECT_EQ(2, smatrix->columns);
    EXPECT_EQ(2, smatrix->rows);
    
    smatrix = oap::host::GetSubMatrix (matrix, 3, 3, smatrix);
    EXPECT_NE(smatrix, submatrix);

    EXPECT_EQ(1, smatrix->columns);
    EXPECT_EQ(1, smatrix->rows);

    EXPECT_EQ(33, smatrix->reValues[0]);

    oap::host::DeleteMatrix (smatrix);
  }
}


