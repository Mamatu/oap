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
#include "gtest/gtest.h"

class OapArnoldiPackageCallbackTests : public testing::Test {
  public:

    virtual void SetUp() {}
    virtual void TearDown() {}

    void executeArnoldiTest (floatt value, const std::string& path,
                             uintt hdim = 32, bool enableValidateOfV = true,
                             floatt tolerance = 0.01)
    {
      oap::ACTestData data(path);

      using UserPair = std::pair<oap::ACTestData*, bool>;

      UserPair userPair = std::make_pair(&data, enableValidateOfV);

      auto multiply = [&userPair] (math::ComplexMatrix* w, math::ComplexMatrix* v, oap::HostProcedures& cuProceduresApi, oap::VecMultiplicationType mt)
      {
        if (mt == oap::VecMultiplicationType::TYPE_WV) {
          oap::ACTestData* data = userPair.first;
          data->load();
          oap::host::CopyHostMatrixToHostMatrix(data->hostV, v);
          if (userPair.second) {
            ASSERT_THAT(data->hostV,
                        MatrixIsEqual(
                            data->refV,
                            InfoType(InfoType::MEAN | InfoType::LARGEST_DIFF)));
          }
          oap::host::CopyHostMatrixToHostMatrix(w, data->refW);
        }
      };

      oap::generic::CuHArnoldiS ca;
      oap::HostProcedures hp;
      math::MatrixInfo matrixInfo (true, true, data.getElementsCount(), data.getElementsCount());

      oap::generic::allocStage1 (ca, matrixInfo, oap::host::NewHostMatrix);
      oap::generic::allocStage2 (ca, matrixInfo, 32, oap::host::NewHostMatrix, oap::host::NewHostMatrix);
      oap::generic::allocStage3 (ca, matrixInfo, 32, oap::host::NewHostMatrix, oap::QRType::QRGR);

      oap::generic::iram_executeInit (ca, hp, multiply);

      oap::generic::deallocStage1 (ca, oap::host::DeleteMatrix);
      oap::generic::deallocStage2 (ca, oap::host::DeleteMatrix, oap::host::DeleteMatrix);
      oap::generic::deallocStage3 (ca, oap::host::DeleteMatrix);
    }
};

TEST_F(OapArnoldiPackageCallbackTests, TestData1) {
  executeArnoldiTest(-3.25, "data/data1");
}

TEST_F(OapArnoldiPackageCallbackTests, TestData2Dim32x32) {
  executeArnoldiTest(-4.257104, "data/data2", 32);
}

TEST_F(OapArnoldiPackageCallbackTests, DISABLED_TestData2Dim64x64) {
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
