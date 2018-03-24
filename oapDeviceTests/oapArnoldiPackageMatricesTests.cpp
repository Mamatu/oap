/*
 * Copyright 2016, 2017 Marcin Matula
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

#include <algorithm>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "gtest/gtest.h"

#include "ArnoldiProceduresImpl.h"
#include "MatchersUtils.h"

#include "Config.h"
#include "oapCudaMatrixUtils.h"
#include "KernelExecutor.h"
#include "oapHostMatrixUtils.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapDeviceMatrixPtr.h"
#include "oapHostMatrixPtr.h"

#include "SmsDataCollector.h"
#include "InfoType.h"

using GetValue = std::function<floatt(size_t xy)>;
using HostMatrixPtrs = std::vector<oap::HostMatrixPtr>;

class OapArnoldiPackageMatricesTests : public testing::Test {
  public:
    CuHArnoldiCallback* m_arnoldiCuda;

    virtual void SetUp() {
      oap::cuda::Context::Instance().create();
      m_arnoldiCuda = new CuHArnoldiCallback();
    }

    virtual void TearDown() {
      delete m_arnoldiCuda;
      oap::cuda::Context::Instance().destroy();
    }

    static void multiply(math::Matrix* m_w, math::Matrix* m_v,
        oap::CuProceduresApi& cuProceduresApi,
        void* userData, CuHArnoldi::MultiplicationType mt)
    {
      UserData* userDataObj = static_cast<UserData*>(userData);

      math::Matrix* hmatrix = userDataObj->hmatrix;
      math::Matrix* hvectorT = userDataObj->hvectorT;
      math::Matrix* dvectorT = userDataObj->dvectorT;
      math::Matrix* dvalue = userDataObj->dvalue;

      for (size_t idx = 0; idx < hmatrix->rows; ++idx) {
        oap::host::GetTransposeReVector(hvectorT, hmatrix, idx);
        //oap::host::PrintMatrix("hvectorT = ", hvectorT);
        oap::cuda::CopyHostMatrixToDeviceMatrix(dvectorT, hvectorT);
        cuProceduresApi.dotProduct(dvalue, dvectorT, m_v);
        oap::cuda::SetReMatrix(m_w, dvalue, 0, idx);
      }
      //oap::cuda::PrintMatrix("m_w =", m_w);
      //oap::cuda::PrintMatrix("m_v =", m_v);
    }

    static bool check(floatt reevalue, floatt imevalue, math::Matrix* vector, uint index, uint max, void* userData) {
      CheckUserData* checkUserData = static_cast<CheckUserData*>(userData);

      oap::HostMatrixPtr eigenvectors = checkUserData->eigenvectors;
      oap::HostMatrixPtr tmpVector = checkUserData->tmpVector;

      const std::vector<EigenPair>* eigenPairs = checkUserData->eigenPairs;

      EigenPair eigenPair = (*eigenPairs)[index];
      uint vectorIndex = eigenPair.getIndex();

      oap::host::GetVector(tmpVector, eigenvectors, vectorIndex);

      return true;
    }

    oap::HostMatrixPtr createSquareMatrix(size_t size, GetValue getValue)
    {
        oap::HostMatrixPtr hmatrix = oap::host::NewMatrix(size, size, 0);

        for (size_t xy = 0; xy < size; ++xy) {
          hmatrix->reValues[GetIndex(hmatrix, xy, xy)] = getValue(xy);
        }

        return hmatrix;
    }

    oap::HostMatrixPtr loadSMSMatrix(const std::string& dir) {
      return oap::host::ReadMatrix(dir + "/smsmatrix.matrix");
    }

    oap::HostMatrixPtr loadEigenvaluesMatrix(const std::string& dir) {
      return oap::host::ReadMatrix(dir + "/eigenvalues.matrix");
    }

    std::vector<EigenPair> getEigenvalues(const std::vector<EigenPair>& avalues, uint wanted) {

      std::vector<EigenPair> values(avalues);

      std::sort(values.begin(), values.end(), ArnUtils::SortLargestReValues);

      std::vector<EigenPair> output(values.begin(), values.begin() + wanted);

      return output;
    }

    oap::HostMatrixPtr loadEigenvector(const std::string& dir, uint index) {
      std::string filename = "eigenvector.matrix";
      filename += std::to_string(index);
      return oap::host::ReadMatrix(dir + "/" + filename);
    }

    HostMatrixPtrs loadMatrices(const std::string& dir) {
      HostMatrixPtrs ptrs;

      oap::HostMatrixPtr smsMatrix = loadSMSMatrix(dir);
      oap::HostMatrixPtr eigenvalues = loadEigenvaluesMatrix(dir);

      ptrs.push_back(smsMatrix);
      ptrs.push_back(eigenvalues);

      for (uint index = 0; index < eigenvalues->columns; ++index) {
        ptrs.push_back(loadEigenvector(dir, index));     
      }

      return ptrs;
    }

    struct UserData {
      oap::HostMatrixPtr hmatrix;
      oap::HostMatrixPtr hvectorT;
      oap::DeviceMatrixPtr dvectorT;
      oap::DeviceMatrixPtr dvalue;
    };

    struct CheckUserData {
      const std::vector<EigenPair>* eigenPairs;
      oap::HostMatrixPtr eigenvectors;
      oap::HostMatrixPtr tmpVector;
    };

    void runMatrixTest(oap::HostMatrixPtr hmatrix, oap::HostMatrixPtr eigenvectors, const std::vector<EigenPair>& eigenPairs, uint hdim, floatt tolerance)
    {
      debugLongTest();

      UserData userData = {
              hmatrix,
              oap::host::NewReMatrix(hmatrix->columns, 1),
              oap::cuda::NewDeviceReMatrix(hmatrix->columns, 1),
              oap::cuda::NewDeviceReMatrix(1, 1)
      };

      CheckUserData checkUserData = {
              &eigenPairs,
              eigenvectors,
              oap::host::NewReMatrix(1, hmatrix->rows)
      };

      m_arnoldiCuda->setOutputType(ArnUtils::HOST);
      m_arnoldiCuda->setCalcTraingularHType(ArnUtils::CALC_IN_HOST);

      m_arnoldiCuda->setCallback(multiply, &userData);
      m_arnoldiCuda->setRho(1. / 3.14159265359);
      m_arnoldiCuda->setSortType(ArnUtils::SortLargestReValues);
      m_arnoldiCuda->setCheckType(ArnUtils::CHECK_INTERNAL);

      m_arnoldiCuda->setCheckCallback(check, &checkUserData);

      uint wanted = eigenPairs.size();

      std::unique_ptr<floatt[]> revalues(new floatt[wanted]);

      std::vector<math::Matrix*> revectors;

      for (size_t idx = 0; idx < wanted; ++idx) {
        revectors.push_back(oap::host::NewReMatrix(1, hmatrix->rows));
      }

      oap::HostMatricesPtr revectorsPtr = oap::makeHostMatricesPtr(revectors);

      m_arnoldiCuda->setOutputsEigenvalues(revalues.get(), NULL);
      m_arnoldiCuda->setOutputsEigenvectors(revectorsPtr);

      math::MatrixInfo matrixInfo(hmatrix);

      m_arnoldiCuda->execute(hdim, wanted, matrixInfo);


      std::vector<floatt> outputValues(&revalues[0], &revalues[wanted]);

      std::vector<floatt> expectedValues;
      for (uint idx = 0; idx < eigenPairs.size(); ++idx) {
        expectedValues.push_back(eigenPairs[idx].re());
      }

      oap::HostMatrixPtr expected = oap::host::NewReMatrix(1, eigenvectors->rows);
      auto compareEigenVector = [&revectorsPtr, expected](math::Matrix* expecteds, uint index)
      {
        math::Matrix* actual = revectorsPtr[index];
        oap::host::GetVector(expected.get(), expecteds, index);;
        EXPECT_THAT(expected.get(), MatrixIsEqual(actual, InfoType::MEAN));
      };

      for (uint index = 0; index < wanted; ++index)
      {
        floatt outcome = m_arnoldiCuda->testOutcome(index);
        EXPECT_LE (outcome, tolerance);
      }
    }

    void runSmsDataTest(uint index, uint wanted, uint hdim = 32, floatt tolerance = 0) {
      uint columns = smsdata_columns[index];
      uint rows = smsdata_rows[index];

      double* smsmatrix = smsdata_matrices[index];
      double* eigenvalues = smsdata_eigenvalues[index];
      double* eigenvectors = smsdata_eigenvectors[index];

      oap::HostMatrixPtr hmatrix = oap::host::NewMatrixCopy<double>(columns, rows, smsmatrix, NULL);
      oap::HostMatrixPtr evmatrix = oap::host::NewMatrixCopy<double>(columns, rows, eigenvectors, NULL);

      std::vector<double> ev(eigenvalues, eigenvalues + columns);
      std::vector<EigenPair> eigenPairs;

      for (uint columnIdx = 0; columnIdx < columns; ++columnIdx) {
        eigenPairs.push_back(EigenPair(ev[columnIdx], columnIdx));
      }

      runMatrixTest(hmatrix, evmatrix, getEigenvalues(eigenPairs, wanted), hdim, tolerance);
    }
};

TEST_F(OapArnoldiPackageMatricesTests, Sms1HeaderTest) {
  runSmsDataTest(0, 5, 15, 0.02);
}

TEST_F(OapArnoldiPackageMatricesTests, Sms2HeaderTest) {
  runSmsDataTest(1, 5, 20, 0.047);
}

TEST_F(OapArnoldiPackageMatricesTests, Sms3HeaderTest) {
  runSmsDataTest(2, 5, 22, 0.1);
}

TEST_F(OapArnoldiPackageMatricesTests, Sms4HeaderTest) {
  runSmsDataTest(3, 6, 14, 0.13);
}

