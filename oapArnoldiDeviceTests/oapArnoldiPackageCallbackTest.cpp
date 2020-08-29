/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "oapTestDataLoader.h"
#include "matrix1.h"
#include "matrix2.h"
#include "matrix3.h"
#include "matrix4.h"
#include "matrix5.h"
#include "gtest/gtest.h"

#include "ArnoldiProceduresImpl.h"
#include "MathOperationsCpu.h"

class OapArnoldiPackageCallbackTests : public testing::Test {
  public:
    CuHArnoldiCallback* arnoldiCuda;

    virtual void SetUp() {
      oap::cuda::Context::Instance().create();

      arnoldiCuda = new CuHArnoldiCallback();
      arnoldiCuda->setOutputType (ArnUtils::HOST);
      arnoldiCuda->setQRType (oap::QRType::QRGR);
      arnoldiCuda->setVecInitType (oap::InitVVectorType::FIRST_VALUE_IS_ONE);
    }

    virtual void TearDown() {
      delete arnoldiCuda;
      oap::cuda::Context::Instance().destroy();
    }

    void executeArnoldiTest(floatt value, const std::string& path,
                            uintt hdim = 32, bool enableValidateOfV = true,
                            floatt tolerance = 0.01) {
      oap::ACTestData data(path);

      typedef std::pair<oap::ACTestData*, bool> UserPair;

      UserPair userPair = std::make_pair(&data, enableValidateOfV);

      class MultiplyFunc {
       public:
        static void multiply (math::Matrix* w, math::Matrix* v, oap::CuProceduresApi& cuProceduresApi,
                              void* userData, oap::VecMultiplicationType mt)
        {
          if (mt == oap::VecMultiplicationType::TYPE_WV) {
            UserPair* userPair = static_cast<UserPair*>(userData);
            oap::ACTestData* data = userPair->first;
            data->load();
            oap::cuda::CopyDeviceMatrixToHostMatrix(data->hostV, v);
            if (userPair->second) {
              ASSERT_THAT(data->hostV,
                          MatrixIsEqual(
                              data->refV,
                              InfoType(InfoType::MEAN | InfoType::LARGEST_DIFF)));
            }
            oap::cuda::CopyHostMatrixToDeviceMatrix(w, data->refW);
          }
        }
      };
      floatt revalues[2] = {0, 0};
      floatt imvalues[2] = {0, 0};

      uintt wanted = 1;

      arnoldiCuda->setCallback(MultiplyFunc::multiply, &userPair);
      arnoldiCuda->setBLimit(0.01);
      arnoldiCuda->setRho(1. / 3.14159265359);
      arnoldiCuda->setSortType(ArnUtils::SortSmallestReValues);
      arnoldiCuda->setCheckType(ArnUtils::CHECK_FIRST_STOP);
      arnoldiCuda->setOutputsEigenvalues(revalues, imvalues);
      arnoldiCuda->setVecInitType (oap::InitVVectorType::FIRST_VALUE_IS_ONE);
      math::MatrixInfo matrixInfo(true, true, data.getElementsCount(),
                                      data.getElementsCount());

      logInfoLongTest();

      arnoldiCuda->execute(hdim, wanted, matrixInfo);
      EXPECT_THAT(revalues[0], ::testing::DoubleNear(value, tolerance));
      // EXPECT_DOUBLE_EQ(value, );
      EXPECT_DOUBLE_EQ(revalues[1], 0);
      EXPECT_DOUBLE_EQ(imvalues[0], 0);
      EXPECT_DOUBLE_EQ(imvalues[1], 0);
    }

    void triangularityTest(const std::string& matrixStr) {
      math::Matrix* matrix = oap::host::NewMatrix(matrixStr);
      triangularityTest(matrix);
      oap::host::DeleteMatrix(matrix);
    }

    void triangularityTest(const math::Matrix* matrix) {
      floatt limit = 0.001;
      for (int fa = 0; fa < gColumns (matrix) - 1; ++fa) {
        floatt value = GetRe (matrix, fa + 1, fa);
        bool islower = value < limit;
        bool isgreater = -limit < value;
        EXPECT_TRUE(islower) << value << " is greater than " << limit
                             << " index= " << fa << ", " << fa + 1;
        EXPECT_TRUE(isgreater) << value << " is lower than " << -limit
                               << " index= " << fa << ", " << fa + 1;
      }
    }
};

TEST_F(OapArnoldiPackageCallbackTests, MagnitudeTest) {
  oap::ACTestData data("data/data1");
  data.load();

  bool isre = data.refW->re.ptr != NULL;
  bool isim = data.refW->im.ptr != NULL;

  uintt columns = gColumns (data.refW);
  uintt rows = gRows (data.refW);

  math::Matrix* dmatrix = oap::cuda::NewDeviceMatrix(isre, isim, columns, rows);

  oap::cuda::CopyHostMatrixToDeviceMatrix(dmatrix, data.refW);

  oap::CuProceduresApi cuProceduresApi;

  floatt output = -1;
  floatt doutput = -1;
  cuProceduresApi.magnitude(doutput, dmatrix);

  math::MathOperationsCpu mocpu;
  mocpu.magnitude(&output, data.refW);

  EXPECT_DOUBLE_EQ(3.25, output);
  EXPECT_DOUBLE_EQ(3.25, doutput);
  EXPECT_DOUBLE_EQ(output, doutput);

  oap::cuda::DeleteDeviceMatrix(dmatrix);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData1) {
  executeArnoldiTest(-3.25, "data/data1");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData2Dim32x32) {
  executeArnoldiTest(-4.257104, "data/data2", 32);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData2Dim64x64) {
  executeArnoldiTest(-4.257104, "data/data2", 64, false);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData3Dim32x32) {
  executeArnoldiTest(-5.519614, "data/data3", 32);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData3Dim64x64) {
  executeArnoldiTest(-5.519614, "data/data3", 64, false);
}

TEST_F(OapArnoldiPackageCallbackTests, TestData4) {
  executeArnoldiTest(-6.976581, "data/data4");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData5) {
  executeArnoldiTest(-8.503910, "data/data5");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData6) {
  executeArnoldiTest(-10.064733, "data/data6");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData7) {
  executeArnoldiTest(-13.235305, "data/data7");
}
