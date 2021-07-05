#include "oapHostComplexMatrixApi.hpp"
#include "MatchersUtils.hpp"
#include "Config.hpp"

#include "oapHostComplexMatrixUPtr.hpp"
#include "oapHostComplexMatrixPtr.hpp"
#include <functional>

class OapHostMatrixUtilsTests : public testing::Test {
 public:
  OapHostMatrixUtilsTests() {}

  virtual ~OapHostMatrixUtilsTests() {}

  virtual void SetUp() {}

  virtual void TearDown() {}

  using GetValue = std::function<floatt(uintt, uintt, uintt)>;

  static const std::string testfilepath;

  static math::ComplexMatrix* createMatrix(uintt columns, uintt rows, GetValue getValue) {
    math::ComplexMatrix* m1 = oap::chost::NewComplexMatrixWithValue (true, true, columns, rows, 0);

    for (int idx1 = 0; idx1 < columns; ++idx1) {
      for (int idx2 = 0; idx2 < rows; ++idx2) {
        uintt idx3 = idx1 + idx2 * columns;
        *GetRePtrIndex (m1, idx3) = getValue(idx1, idx2, idx3);
        *GetImPtrIndex (m1, idx3) = getValue(idx1, idx2, idx3);
      }
    }

    return m1;
  }
};

const std::string OapHostMatrixUtilsTests::testfilepath = oap::utils::Config::getFileInTmp("host_tests/test_file");

TEST_F(OapHostMatrixUtilsTests, Copy) {
  const uintt columns = 11;
  const uintt rows = 15;
  math::ComplexMatrix* m1 = oap::chost::NewReMatrixWithValue (columns, rows, 1);
  math::ComplexMatrix* m2 = oap::chost::NewReMatrixWithValue (columns, rows, 0);

  oap::chost::CopyMatrix(m2, m1);

  EXPECT_THAT(m1, MatrixIsEqual(m2));

  oap::chost::DeleteMatrix(m1);
  oap::chost::DeleteMatrix(m2);
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_1)
{
  const uintt columns = 11;
  const uintt rows = 15;

  oap::HostComplexMatrixPtr m1 = oap::chost::NewReMatrixWithValue (15, 15, 1);
  oap::HostComplexMatrixPtr m2 = oap::chost::NewReMatrixWithValue (5, 6, 0);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::chost::GetMatrixInfo);

  oap::chost::CopyHostMatrixToHostMatrixRegion (m2, {0, 0}, m1, {0, 0, {gColumns(m2), gRows(m2)}});

  EXPECT_THAT(m2.get (), MatrixHasValues(1));
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_2)
{
  const uintt columns = 11;
  const uintt rows = 15;

  oap::HostComplexMatrixPtr m1 = oap::chost::NewReMatrixWithValue (15, 15, 1);
  oap::HostComplexMatrixPtr m2 = oap::chost::NewReMatrixWithValue (5, 6, 0);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::chost::GetMatrixInfo);

  oap::generic::setRows (1, dims[oap::generic::g_dstIdx]);
  oap::generic::setColumns (1, dims[oap::generic::g_dstIdx]);

  oap::generic::setRows (1, dims[oap::generic::g_srcIdx]);
  oap::generic::setColumns (1, dims[oap::generic::g_srcIdx]);

  oap::chost::CopyHostMatrixToHostMatrixRegion (m2, {0, 0}, m1, {{0, 0},{1, 1}});

  EXPECT_DOUBLE_EQ(1, GetReIndex (m2, 0));
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_3)
{
  oap::HostComplexMatrixPtr m1 = oap::chost::NewReMatrixWithValue (15, 15, 1);
  oap::HostComplexMatrixPtr m2 = oap::chost::NewReMatrixWithValue (5, 6, 0);
  oap::HostComplexMatrixPtr expected = oap::chost::NewReMatrixWithValue (5, 6, 0);

  std::vector<floatt> expectedValues =
  {
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,1,1,1,
    0,0,1,1,1,
    0,0,1,1,1,
    0,0,1,1,1,
  };
  oap::chost::SetReValuesToMatrix(expected, expectedValues);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::chost::GetMatrixInfo);

  oap::generic::setColumns (3, dims[oap::generic::g_dstIdx]);
  oap::generic::setRows (4, dims[oap::generic::g_dstIdx]);

  oap::generic::setColumns (3, dims[oap::generic::g_srcIdx]);
  oap::generic::setRows (4, dims[oap::generic::g_srcIdx]);

  oap::generic::setColumnIdx (2, dims[oap::generic::g_dstIdx]);
  oap::generic::setRowIdx (2, dims[oap::generic::g_dstIdx]);

  oap::chost::CopyHostMatrixToHostMatrixRegion (m2, {2, 2}, m1, {{0, 0}, {3, 4}});

  EXPECT_THAT(expected.get (), MatrixIsEqual(m2.get())) << "actual: " << oap::chost::to_string(m2);
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_4)
{
  oap::HostComplexMatrixPtr m1 = oap::chost::NewReMatrixWithValue (15, 15, 1);
  oap::HostComplexMatrixPtr m2 = oap::chost::NewReMatrixWithValue (5, 6, 0);
  oap::HostComplexMatrixPtr expected = oap::chost::NewReMatrixWithValue (5, 6, 0);

  std::vector<floatt> expectedValues =
  {
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,1,1,0,
    0,0,1,1,0,
    0,0,1,1,0,
    0,0,0,0,0,
  };
  oap::chost::SetReValuesToMatrix(expected, expectedValues);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::chost::GetMatrixInfo);

  oap::generic::setColumns (2, dims[oap::generic::g_dstIdx]);
  oap::generic::setRows (3, dims[oap::generic::g_dstIdx]);

  oap::generic::setColumns (2, dims[oap::generic::g_srcIdx]);
  oap::generic::setRows (3, dims[oap::generic::g_srcIdx]);

  oap::generic::setColumnIdx (2, dims[oap::generic::g_dstIdx]);
  oap::generic::setRowIdx (2, dims[oap::generic::g_dstIdx]);

  oap::chost::CopyHostMatrixToHostMatrixRegion (m2, {2, 2}, m1, {{0, 0}, {2, 3}});

  EXPECT_THAT(expected.get (), MatrixIsEqual(m2.get())) << "actual: " << oap::chost::to_string(m2);
}

TEST_F(OapHostMatrixUtilsTests, WriteReadMatrix) {

  math::ComplexMatrix* m1 = createMatrix(10, 10, [](uintt, uintt, uintt idx) { return idx; });

  bool status = oap::chost::WriteMatrix(testfilepath, m1);

  EXPECT_TRUE(status);

  if (status) {
    math::ComplexMatrix* m2 = oap::chost::ReadMatrix(testfilepath);

    EXPECT_EQ(gColumns (m2), gColumns (m1));
    EXPECT_EQ(gRows (m2), gRows (m1));

    for (int fa = 0; fa < gColumns (m1) * gRows (m1); ++fa) {
      EXPECT_EQ(GetReIndex (m1, fa), GetReIndex (m2, fa));
      EXPECT_EQ(GetImIndex (m1, fa), GetImIndex (m2, fa));
    }

    oap::chost::DeleteMatrix(m2);
  }
  oap::chost::DeleteMatrix(m1);
}

#if 0
TEST_F(OapHostMatrixUtilsTests, WriteReadMatrixEx) {

  math::ComplexMatrix* m1 = createMatrix(10, 10, [](uintt idx, uintt, uintt) { return idx; });

  bool status = oap::chost::WriteMatrix(testfilepath, m1);

  EXPECT_TRUE(status);

  if (status) {
    MatrixEx mex;
    mex.row = 0;
    gRows (&mex) = 10;

    mex.column = 4;
    gColumns (&mex) = 6;

    math::ComplexMatrix* m2 = oap::chost::ReadMatrix(testfilepath, mex);

    EXPECT_EQ(gColumns (m2), gColumns (&mex));
    EXPECT_EQ(gRows (m2), gRows (&mex));

    for (int fa = 0; fa < gColumns (m2); ++fa)
    {
      for (int fb = 0; fb < gRows (m2); ++fb)
      {
        int idx = fa + gColumns (m2) * fb;
        int idx1 = (mex.column + fa) + gColumns (m1) * (mex.row + fb);
        EXPECT_EQ(GetReIndex (m2, idx], m1->re.mem[idx1));
        EXPECT_EQ(GetImIndex (m2, idx], m1->im.mem[idx1));
      }
    }

    oap::chost::DeleteMatrix(m2);
  }
  oap::chost::DeleteMatrix(m1);
}
#endif

TEST_F(OapHostMatrixUtilsTests, WriteMatrixReadVector) {

  math::ComplexMatrix* matrix = createMatrix(10, 10, [](uintt, uintt, uintt idx) { return idx; });

  bool status = oap::chost::WriteMatrix(testfilepath, matrix);

  EXPECT_TRUE(status);

  if (status) {
    size_t index = 1;
    math::ComplexMatrix* vec = oap::chost::ReadRowVector(testfilepath, index);

    EXPECT_EQ(gColumns (vec), gColumns (matrix));
    EXPECT_EQ(gRows (vec), 1);

    for (int fa = 0; fa < gColumns (matrix); ++fa) {
      EXPECT_EQ(fa + index * gColumns (matrix), GetReIndex (vec, fa));
      EXPECT_EQ(fa + index * gColumns (matrix), GetImIndex (vec, fa));
    }

    oap::chost::DeleteMatrix(vec);
  }
  oap::chost::DeleteMatrix(matrix);
}

TEST_F(OapHostMatrixUtilsTests, NewSubMatrixTests)
{
  auto initMatrix = [](math::ComplexMatrix* matrix)
  {
    for (uintt idx = 0; idx < gColumns (matrix); ++idx)
    {
      for (uintt idx1 = 0; idx1 < gRows (matrix); ++idx1)
      {
        oap::chost::SetReValue (matrix, idx, idx1, idx + 10 * idx1);
        oap::chost::SetImValue (matrix, idx, idx1, idx + 10 * idx1);
      }
    }
  };

  {
    oap::HostComplexMatrixUPtr matrix = oap::chost::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostComplexMatrixUPtr submatrix = oap::chost::NewSubMatrix (matrix, 1, 1, 2, 2);
    EXPECT_EQ(11, GetReIndex (submatrix, 0));
    EXPECT_EQ(12, GetReIndex (submatrix, 1));
    EXPECT_EQ(21, GetReIndex (submatrix, 2));
    EXPECT_EQ(22, GetReIndex (submatrix, 3));
    EXPECT_EQ(2, gColumns (submatrix));
    EXPECT_EQ(2, gRows (submatrix));
  }

  {
    oap::HostComplexMatrixUPtr matrix = oap::chost::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostComplexMatrixUPtr submatrix = oap::chost::NewSubMatrix (matrix, 1, 1, 6, 6);
    EXPECT_EQ(11, GetReIndex (submatrix, 0));
    EXPECT_EQ(12, GetReIndex (submatrix, 1));
    EXPECT_EQ(13, GetReIndex (submatrix, 2));
    EXPECT_EQ(21, GetReIndex (submatrix, 3));
    EXPECT_EQ(22, GetReIndex (submatrix, 4));
    EXPECT_EQ(23, GetReIndex (submatrix, 5));
    EXPECT_EQ(3, gColumns (submatrix));
    EXPECT_EQ(3, gRows (submatrix));
  }

  {
    oap::HostComplexMatrixUPtr matrix = oap::chost::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostComplexMatrixUPtr submatrix = oap::chost::NewSubMatrix (matrix, 0, 0, 4, 4);

    EXPECT_EQ(0, oap::chost::GetReValue(submatrix, 0, 0));
    EXPECT_EQ(1, oap::chost::GetReValue(submatrix, 1, 0));
    EXPECT_EQ(2, oap::chost::GetReValue(submatrix, 2, 0));
    EXPECT_EQ(3, oap::chost::GetReValue(submatrix, 3, 0));

    EXPECT_EQ(10, oap::chost::GetReValue(submatrix, 0, 1));
    EXPECT_EQ(11, oap::chost::GetReValue(submatrix, 1, 1));
    EXPECT_EQ(12, oap::chost::GetReValue(submatrix, 2, 1));
    EXPECT_EQ(13, oap::chost::GetReValue(submatrix, 3, 1));

    EXPECT_EQ(20,oap::chost::GetReValue(submatrix, 0, 2));
    EXPECT_EQ(21,oap::chost::GetReValue(submatrix, 1, 2));
    EXPECT_EQ(22,oap::chost::GetReValue(submatrix, 2, 2));
    EXPECT_EQ(23,oap::chost::GetReValue(submatrix, 3, 2));

    EXPECT_EQ(30,oap::chost::GetReValue(submatrix, 0, 3));
    EXPECT_EQ(31,oap::chost::GetReValue(submatrix, 1, 3));
    EXPECT_EQ(32,oap::chost::GetReValue(submatrix, 2, 3));
    EXPECT_EQ(33,oap::chost::GetReValue(submatrix, 3, 3));
  }
}

TEST_F(OapHostMatrixUtilsTests, GetSubMatrixTest)
{
  auto initMatrix = [](math::ComplexMatrix* matrix)
  {
    for (uintt idx = 0; idx < gColumns (matrix); ++idx)
    {
      for (uintt idx1 = 0; idx1 < gRows (matrix); ++idx1)
      {
        oap::chost::SetReValue (matrix, idx, idx1, idx + 10 * idx1);
        oap::chost::SetImValue (matrix, idx, idx1, idx + 10 * idx1);
      }
    }
  };

  {
    oap::HostComplexMatrixUPtr matrix = oap::chost::NewReMatrix(4, 4);
    initMatrix (matrix);
    math::ComplexMatrix* submatrix = oap::chost::NewSubMatrix (matrix, 1, 1, 2, 2);

    EXPECT_EQ(2, gColumns (submatrix));
    EXPECT_EQ(2, gRows (submatrix));

    EXPECT_EQ(11, GetReIndex (submatrix, 0));
    EXPECT_EQ(12, GetReIndex (submatrix, 1));
    EXPECT_EQ(21, GetReIndex (submatrix, 2));
    EXPECT_EQ(22, GetReIndex (submatrix, 3));

    math::ComplexMatrix* smatrix = oap::chost::GetSubMatrix (matrix, 0, 0, submatrix);
    EXPECT_EQ(smatrix, submatrix);

    EXPECT_EQ(2, gColumns (smatrix));
    EXPECT_EQ(2, gRows (smatrix));

    EXPECT_EQ(0, GetReIndex (smatrix, 0));
    EXPECT_EQ(1, GetReIndex (smatrix, 1));
    EXPECT_EQ(10, GetReIndex (smatrix, 2));
    EXPECT_EQ(11, GetReIndex (smatrix, 3));

    smatrix = oap::chost::GetSubMatrix (matrix, 1, 1, smatrix);
    EXPECT_EQ(smatrix, submatrix);

    EXPECT_EQ(2, gColumns (smatrix));
    EXPECT_EQ(2, gRows (smatrix));

    EXPECT_EQ(11, GetReIndex (smatrix, 0));
    EXPECT_EQ(12, GetReIndex (smatrix, 1));
    EXPECT_EQ(21, GetReIndex (smatrix, 2));
    EXPECT_EQ(22, GetReIndex (smatrix, 3));

    smatrix = oap::chost::GetSubMatrix (matrix, 2, 2, smatrix);
    EXPECT_EQ(smatrix, submatrix);
    EXPECT_EQ(22, GetReIndex (smatrix, 0));
    EXPECT_EQ(23, GetReIndex (smatrix, 1));
    EXPECT_EQ(32, GetReIndex (smatrix, 2));
    EXPECT_EQ(33, GetReIndex (smatrix, 3));
    EXPECT_EQ(2, gColumns (smatrix));
    EXPECT_EQ(2, gRows (smatrix));

    math::MatrixInfo submatrixInfo = oap::chost::GetMatrixInfo (submatrix);

    smatrix = oap::chost::GetSubMatrix (matrix, 3, 3, smatrix);

    math::MatrixInfo smatrixInfo = oap::chost::GetMatrixInfo (smatrix);

    EXPECT_NE(smatrixInfo, submatrixInfo);

    EXPECT_EQ(1, gColumns (smatrix));
    EXPECT_EQ(1, gRows (smatrix));

    EXPECT_EQ(33, GetReIndex (smatrix, 0));

    oap::chost::DeleteMatrix (smatrix);
  }
}

TEST_F(OapHostMatrixUtilsTests, SetZeroRow_1)
{
  const uintt rows = 10;
  const uintt columns = 10;
  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewComplexMatrixWithValue (columns, rows, 1.f);

  oap::chost::SetZeroRow (hostMatrix.get(), 0);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      if (x == 0)
      {
        EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      }
      else
      {
        EXPECT_EQ (1.f, hostMatrix->re.mem.ptr[x + columns * y]);
      }
    }
  }
  printf ("%s\n", oap::chost::to_string(hostMatrix.get()).c_str());
}

TEST_F(OapHostMatrixUtilsTests, SetZeroRow_2)
{
  const uintt rows = 10;
  const uintt columns = 10;
  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewComplexMatrixWithValue (columns, rows, 1.f);

  oap::chost::SetZeroRow (hostMatrix, 1);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      if (x == 1)
      {
        EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      }
      else
      {
        EXPECT_EQ (1.f, hostMatrix->re.mem.ptr[x + columns * y]);
      }
    }
  }
  printf ("%s\n", oap::chost::to_string(hostMatrix.get()).c_str());
}

TEST_F(OapHostMatrixUtilsTests, SetZeroMatrix_1)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewComplexMatrixWithValue (columns, rows, 1.f);

  oap::chost::SetZeroMatrix (hostMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      EXPECT_EQ (0, hostMatrix->im.mem.ptr[x + columns * y]);
    }
  }
}

TEST_F(OapHostMatrixUtilsTests, SetZeroMatrix_2)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewReMatrixWithValue (columns, rows, 1.f);

  oap::chost::SetZeroMatrix (hostMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      EXPECT_EQ (nullptr, hostMatrix->im.mem.ptr);
    }
  }
}

TEST_F(OapHostMatrixUtilsTests, SetZeroMatrix_3)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewImMatrixWithValue (columns, rows, 1.f);

  oap::chost::SetZeroMatrix (hostMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (nullptr, hostMatrix->re.mem.ptr);
      EXPECT_EQ (0, hostMatrix->im.mem.ptr[x + columns * y]);
    }
  }
}

TEST_F(OapHostMatrixUtilsTests, SetZeroMatrix_4)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewReMatrixWithValue (columns, rows, 1.f);

  oap::chost::SetZeroReMatrix (hostMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (0, hostMatrix->re.mem.ptr[x + columns * y]);
      EXPECT_EQ (nullptr, hostMatrix->im.mem.ptr);
    }
  }
}

TEST_F(OapHostMatrixUtilsTests, SetZeroMatrix_5)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewImMatrixWithValue (columns, rows, 1.f);

  oap::chost::SetZeroImMatrix (hostMatrix);

  EXPECT_EQ (rows, gRows (hostMatrix));
  EXPECT_EQ (columns, gColumns (hostMatrix));
  for (uintt y = 0; y < rows; ++y)
  {
    for (uintt x = 0; x < columns; ++x)
    {
      EXPECT_EQ (nullptr, hostMatrix->re.mem.ptr);
      EXPECT_EQ (0, hostMatrix->im.mem.ptr[x + columns * y]);
    }
  }
}

TEST_F(OapHostMatrixUtilsTests, GetDiagonal_1)
{
  const uintt rows = 4;
  const uintt columns = 4;

  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewReMatrixWithValue (columns, rows, 1.f);
  for (uintt x = 0; x < 4; ++x)
  {
    for (uintt y = 0; y < 4; ++y)
    {
      if (x != y)
      {
        oap::chost::SetReValue (hostMatrix, x, y, 2.);
      }
    }
  }

  EXPECT_DOUBLE_EQ(1.f, oap::chost::GetReDiagonal (hostMatrix, 0));
  EXPECT_DOUBLE_EQ(1.f, oap::chost::GetReDiagonal (hostMatrix, 1));
  EXPECT_DOUBLE_EQ(1.f, oap::chost::GetReDiagonal (hostMatrix, 2));
  EXPECT_DOUBLE_EQ(1.f, oap::chost::GetReDiagonal (hostMatrix, 3));
}

TEST_F(OapHostMatrixUtilsTests, GetDiagonal_2)
{
  const uintt rows = 6;
  const uintt columns = 6;

  oap::HostComplexMatrixUPtr hostMatrix = oap::chost::NewReMatrixWithValue (columns, rows, 10.f);
  for (uintt x = 0; x < 6; ++x)
  {
    for (uintt y = 0; y < 6; ++y)
    {
      if (x == y)
      {
        oap::chost::SetReValue (hostMatrix, x, y, static_cast<floatt>(x));
      }
    }
  }

  EXPECT_DOUBLE_EQ(0.f, oap::chost::GetReDiagonal (hostMatrix, 0));
  EXPECT_DOUBLE_EQ(1.f, oap::chost::GetReDiagonal (hostMatrix, 1));
  EXPECT_DOUBLE_EQ(2.f, oap::chost::GetReDiagonal (hostMatrix, 2));
  EXPECT_DOUBLE_EQ(3.f, oap::chost::GetReDiagonal (hostMatrix, 3));
  EXPECT_DOUBLE_EQ(4.f, oap::chost::GetReDiagonal (hostMatrix, 4));
  EXPECT_DOUBLE_EQ(5.f, oap::chost::GetReDiagonal (hostMatrix, 5));
}
