#include "oapHostMatrixUtils.h"
#include "MatchersUtils.h"
#include "Config.h"

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

  static math::ComplexMatrix* createMatrix(uintt columns, uintt rows, GetValue getValue) {
    math::ComplexMatrix* m1 = oap::host::NewMatrixWithValue (true, true, columns, rows, 0);

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
  math::ComplexMatrix* m1 = oap::host::NewReMatrixWithValue (columns, rows, 1);
  math::ComplexMatrix* m2 = oap::host::NewReMatrixWithValue (columns, rows, 0);

  oap::host::CopyMatrix(m2, m1);

  EXPECT_THAT(m1, MatrixIsEqual(m2));

  oap::host::DeleteMatrix(m1);
  oap::host::DeleteMatrix(m2);
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_1)
{
  const uintt columns = 11;
  const uintt rows = 15;

  oap::HostMatrixPtr m1 = oap::host::NewReMatrixWithValue (15, 15, 1);
  oap::HostMatrixPtr m2 = oap::host::NewReMatrixWithValue (5, 6, 0);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::host::GetMatrixInfo);

  oap::host::CopyHostMatrixToHostMatrixRegion (m2, {0, 0}, m1, {0, 0, {gColumns(m2), gRows(m2)}});

  EXPECT_THAT(m2.get (), MatrixHasValues(1));
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_2)
{
  const uintt columns = 11;
  const uintt rows = 15;

  oap::HostMatrixPtr m1 = oap::host::NewReMatrixWithValue (15, 15, 1);
  oap::HostMatrixPtr m2 = oap::host::NewReMatrixWithValue (5, 6, 0);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::host::GetMatrixInfo);

  oap::generic::setRows (1, dims[oap::generic::g_dstIdx]);
  oap::generic::setColumns (1, dims[oap::generic::g_dstIdx]);

  oap::generic::setRows (1, dims[oap::generic::g_srcIdx]);
  oap::generic::setColumns (1, dims[oap::generic::g_srcIdx]);

  oap::host::CopyHostMatrixToHostMatrixRegion (m2, {0, 0}, m1, {{0, 0},{1, 1}});

  EXPECT_DOUBLE_EQ(1, GetReIndex (m2, 0));
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_3)
{
  oap::HostMatrixPtr m1 = oap::host::NewReMatrixWithValue (15, 15, 1);
  oap::HostMatrixPtr m2 = oap::host::NewReMatrixWithValue (5, 6, 0);
  oap::HostMatrixPtr expected = oap::host::NewReMatrixWithValue (5, 6, 0);

  std::vector<floatt> expectedValues =
  {
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,1,1,1,
    0,0,1,1,1,
    0,0,1,1,1,
    0,0,1,1,1,
  };
  oap::host::SetReValuesToMatrix(expected, expectedValues);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::host::GetMatrixInfo);

  oap::generic::setColumns (3, dims[oap::generic::g_dstIdx]);
  oap::generic::setRows (4, dims[oap::generic::g_dstIdx]);

  oap::generic::setColumns (3, dims[oap::generic::g_srcIdx]);
  oap::generic::setRows (4, dims[oap::generic::g_srcIdx]);

  oap::generic::setColumnIdx (2, dims[oap::generic::g_dstIdx]);
  oap::generic::setRowIdx (2, dims[oap::generic::g_dstIdx]);

  oap::host::CopyHostMatrixToHostMatrixRegion (m2, {2, 2}, m1, {{0, 0}, {3, 4}});

  EXPECT_THAT(expected.get (), MatrixIsEqual(m2.get())) << "actual: " << oap::host::to_string(m2);
}

TEST_F(OapHostMatrixUtilsTests, SubCopyTest_4)
{
  oap::HostMatrixPtr m1 = oap::host::NewReMatrixWithValue (15, 15, 1);
  oap::HostMatrixPtr m2 = oap::host::NewReMatrixWithValue (5, 6, 0);
  oap::HostMatrixPtr expected = oap::host::NewReMatrixWithValue (5, 6, 0);

  std::vector<floatt> expectedValues =
  {
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,1,1,0,
    0,0,1,1,0,
    0,0,1,1,0,
    0,0,0,0,0,
  };
  oap::host::SetReValuesToMatrix(expected, expectedValues);

  uintt dims[2][2][2];
  oap::generic::initDims (dims, m2, oap::host::GetMatrixInfo);

  oap::generic::setColumns (2, dims[oap::generic::g_dstIdx]);
  oap::generic::setRows (3, dims[oap::generic::g_dstIdx]);

  oap::generic::setColumns (2, dims[oap::generic::g_srcIdx]);
  oap::generic::setRows (3, dims[oap::generic::g_srcIdx]);

  oap::generic::setColumnIdx (2, dims[oap::generic::g_dstIdx]);
  oap::generic::setRowIdx (2, dims[oap::generic::g_dstIdx]);

  oap::host::CopyHostMatrixToHostMatrixRegion (m2, {2, 2}, m1, {{0, 0}, {2, 3}});

  EXPECT_THAT(expected.get (), MatrixIsEqual(m2.get())) << "actual: " << oap::host::to_string(m2);
}

TEST_F(OapHostMatrixUtilsTests, WriteReadMatrix) {

  math::ComplexMatrix* m1 = createMatrix(10, 10, [](uintt, uintt, uintt idx) { return idx; });

  bool status = oap::host::WriteMatrix(testfilepath, m1);

  EXPECT_TRUE(status);

  if (status) {
    math::ComplexMatrix* m2 = oap::host::ReadMatrix(testfilepath);

    EXPECT_EQ(gColumns (m2), gColumns (m1));
    EXPECT_EQ(gRows (m2), gRows (m1));

    for (int fa = 0; fa < gColumns (m1) * gRows (m1); ++fa) {
      EXPECT_EQ(GetReIndex (m1, fa), GetReIndex (m2, fa));
      EXPECT_EQ(GetImIndex (m1, fa), GetImIndex (m2, fa));
    }

    oap::host::DeleteMatrix(m2);
  }
  oap::host::DeleteMatrix(m1);
}

#if 0
TEST_F(OapHostMatrixUtilsTests, WriteReadMatrixEx) {

  math::ComplexMatrix* m1 = createMatrix(10, 10, [](uintt idx, uintt, uintt) { return idx; });

  bool status = oap::host::WriteMatrix(testfilepath, m1);

  EXPECT_TRUE(status);

  if (status) {
    MatrixEx mex;
    mex.row = 0;
    gRows (&mex) = 10;

    mex.column = 4;
    gColumns (&mex) = 6;

    math::ComplexMatrix* m2 = oap::host::ReadMatrix(testfilepath, mex);

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

    oap::host::DeleteMatrix(m2);
  }
  oap::host::DeleteMatrix(m1);
}
#endif

TEST_F(OapHostMatrixUtilsTests, WriteMatrixReadVector) {

  math::ComplexMatrix* matrix = createMatrix(10, 10, [](uintt, uintt, uintt idx) { return idx; });

  bool status = oap::host::WriteMatrix(testfilepath, matrix);

  EXPECT_TRUE(status);

  if (status) {
    size_t index = 1;
    math::ComplexMatrix* vec = oap::host::ReadRowVector(testfilepath, index);

    EXPECT_EQ(gColumns (vec), gColumns (matrix));
    EXPECT_EQ(gRows (vec), 1);

    for (int fa = 0; fa < gColumns (matrix); ++fa) {
      EXPECT_EQ(fa + index * gColumns (matrix), GetReIndex (vec, fa));
      EXPECT_EQ(fa + index * gColumns (matrix), GetImIndex (vec, fa));
    }

    oap::host::DeleteMatrix(vec);
  }
  oap::host::DeleteMatrix(matrix);
}

TEST_F(OapHostMatrixUtilsTests, NewSubMatrixTests)
{
  auto initMatrix = [](math::ComplexMatrix* matrix)
  {
    for (uintt idx = 0; idx < gColumns (matrix); ++idx)
    {
      for (uintt idx1 = 0; idx1 < gRows (matrix); ++idx1)
      {
        oap::host::SetReValue (matrix, idx, idx1, idx + 10 * idx1);
        oap::host::SetImValue (matrix, idx, idx1, idx + 10 * idx1);
      }
    }
  };

  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostMatrixUPtr submatrix = oap::host::NewSubMatrix (matrix, 1, 1, 2, 2);
    EXPECT_EQ(11, GetReIndex (submatrix, 0));
    EXPECT_EQ(12, GetReIndex (submatrix, 1));
    EXPECT_EQ(21, GetReIndex (submatrix, 2));
    EXPECT_EQ(22, GetReIndex (submatrix, 3));
    EXPECT_EQ(2, gColumns (submatrix));
    EXPECT_EQ(2, gRows (submatrix));
  }

  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostMatrixUPtr submatrix = oap::host::NewSubMatrix (matrix, 1, 1, 6, 6);
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
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    oap::HostMatrixUPtr submatrix = oap::host::NewSubMatrix (matrix, 0, 0, 4, 4);

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
  auto initMatrix = [](math::ComplexMatrix* matrix)
  {
    for (uintt idx = 0; idx < gColumns (matrix); ++idx)
    {
      for (uintt idx1 = 0; idx1 < gRows (matrix); ++idx1)
      {
        oap::host::SetReValue (matrix, idx, idx1, idx + 10 * idx1);
        oap::host::SetImValue (matrix, idx, idx1, idx + 10 * idx1);
      }
    }
  };

  {
    oap::HostMatrixUPtr matrix = oap::host::NewReMatrix(4, 4);
    initMatrix (matrix);
    math::ComplexMatrix* submatrix = oap::host::NewSubMatrix (matrix, 1, 1, 2, 2);

    EXPECT_EQ(2, gColumns (submatrix));
    EXPECT_EQ(2, gRows (submatrix));

    EXPECT_EQ(11, GetReIndex (submatrix, 0));
    EXPECT_EQ(12, GetReIndex (submatrix, 1));
    EXPECT_EQ(21, GetReIndex (submatrix, 2));
    EXPECT_EQ(22, GetReIndex (submatrix, 3));

    math::ComplexMatrix* smatrix = oap::host::GetSubMatrix (matrix, 0, 0, submatrix);
    EXPECT_EQ(smatrix, submatrix);

    EXPECT_EQ(2, gColumns (smatrix));
    EXPECT_EQ(2, gRows (smatrix));

    EXPECT_EQ(0, GetReIndex (smatrix, 0));
    EXPECT_EQ(1, GetReIndex (smatrix, 1));
    EXPECT_EQ(10, GetReIndex (smatrix, 2));
    EXPECT_EQ(11, GetReIndex (smatrix, 3));

    smatrix = oap::host::GetSubMatrix (matrix, 1, 1, smatrix);
    EXPECT_EQ(smatrix, submatrix);

    EXPECT_EQ(2, gColumns (smatrix));
    EXPECT_EQ(2, gRows (smatrix));

    EXPECT_EQ(11, GetReIndex (smatrix, 0));
    EXPECT_EQ(12, GetReIndex (smatrix, 1));
    EXPECT_EQ(21, GetReIndex (smatrix, 2));
    EXPECT_EQ(22, GetReIndex (smatrix, 3));

    smatrix = oap::host::GetSubMatrix (matrix, 2, 2, smatrix);
    EXPECT_EQ(smatrix, submatrix);
    EXPECT_EQ(22, GetReIndex (smatrix, 0));
    EXPECT_EQ(23, GetReIndex (smatrix, 1));
    EXPECT_EQ(32, GetReIndex (smatrix, 2));
    EXPECT_EQ(33, GetReIndex (smatrix, 3));
    EXPECT_EQ(2, gColumns (smatrix));
    EXPECT_EQ(2, gRows (smatrix));

    math::MatrixInfo submatrixInfo = oap::host::GetMatrixInfo (submatrix);

    smatrix = oap::host::GetSubMatrix (matrix, 3, 3, smatrix);

    math::MatrixInfo smatrixInfo = oap::host::GetMatrixInfo (smatrix);

    EXPECT_NE(smatrixInfo, submatrixInfo);

    EXPECT_EQ(1, gColumns (smatrix));
    EXPECT_EQ(1, gRows (smatrix));

    EXPECT_EQ(33, GetReIndex (smatrix, 0));

    oap::host::DeleteMatrix (smatrix);
  }
}

TEST_F(OapHostMatrixUtilsTests, SetZeroRow_1)
{
  const uintt rows = 10;
  const uintt columns = 10;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 1.f);

  oap::host::SetZeroRow (hostMatrix.get(), 0);

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
  printf ("%s\n", oap::host::to_string(hostMatrix.get()).c_str());
}

TEST_F(OapHostMatrixUtilsTests, SetZeroRow_2)
{
  const uintt rows = 10;
  const uintt columns = 10;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 1.f);

  oap::host::SetZeroRow (hostMatrix, 1);

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
  printf ("%s\n", oap::host::to_string(hostMatrix.get()).c_str());
}

TEST_F(OapHostMatrixUtilsTests, SetZeroMatrix_1)
{
  const uintt rows = 16384;
  const uintt columns = 32;
  oap::HostMatrixUPtr hostMatrix = oap::host::NewMatrixWithValue (columns, rows, 1.f);

  oap::host::SetZeroMatrix (hostMatrix);

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
  oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 1.f);

  oap::host::SetZeroMatrix (hostMatrix);

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
  oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 1.f);

  oap::host::SetZeroMatrix (hostMatrix);

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
  oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 1.f);

  oap::host::SetZeroReMatrix (hostMatrix);

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
  oap::HostMatrixUPtr hostMatrix = oap::host::NewImMatrixWithValue (columns, rows, 1.f);

  oap::host::SetZeroImMatrix (hostMatrix);

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

  oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 1.f);
  for (uintt x = 0; x < 4; ++x)
  {
    for (uintt y = 0; y < 4; ++y)
    {
      if (x != y)
      {
        oap::host::SetReValue (hostMatrix, x, y, 2.);
      }
    }
  }

  EXPECT_DOUBLE_EQ(1.f, oap::host::GetReDiagonal (hostMatrix, 0));
  EXPECT_DOUBLE_EQ(1.f, oap::host::GetReDiagonal (hostMatrix, 1));
  EXPECT_DOUBLE_EQ(1.f, oap::host::GetReDiagonal (hostMatrix, 2));
  EXPECT_DOUBLE_EQ(1.f, oap::host::GetReDiagonal (hostMatrix, 3));
}

TEST_F(OapHostMatrixUtilsTests, GetDiagonal_2)
{
  const uintt rows = 6;
  const uintt columns = 6;

  oap::HostMatrixUPtr hostMatrix = oap::host::NewReMatrixWithValue (columns, rows, 10.f);
  for (uintt x = 0; x < 6; ++x)
  {
    for (uintt y = 0; y < 6; ++y)
    {
      if (x == y)
      {
        oap::host::SetReValue (hostMatrix, x, y, static_cast<floatt>(x));
      }
    }
  }

  EXPECT_DOUBLE_EQ(0.f, oap::host::GetReDiagonal (hostMatrix, 0));
  EXPECT_DOUBLE_EQ(1.f, oap::host::GetReDiagonal (hostMatrix, 1));
  EXPECT_DOUBLE_EQ(2.f, oap::host::GetReDiagonal (hostMatrix, 2));
  EXPECT_DOUBLE_EQ(3.f, oap::host::GetReDiagonal (hostMatrix, 3));
  EXPECT_DOUBLE_EQ(4.f, oap::host::GetReDiagonal (hostMatrix, 4));
  EXPECT_DOUBLE_EQ(5.f, oap::host::GetReDiagonal (hostMatrix, 5));
}
