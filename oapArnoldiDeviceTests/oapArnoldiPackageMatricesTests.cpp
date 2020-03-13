/*
 * Copyright 2016 - 2019 Marcin Matula
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

#include "CMatrixDataCollector.h"
#include "InfoType.h"

using GetValue = std::function<floatt(size_t xy)>;
using HostMatrixPtrs = std::vector<oap::HostMatrixPtr>;

class OapArnoldiPackageMatricesTests : public testing::Test {
  public:
    CuHArnoldiCallback* m_arnoldiCuda;

    virtual void SetUp() {
      oap::cuda::Context::Instance().create();
      m_arnoldiCuda = new CuHArnoldiCallback();
      m_arnoldiCuda->setVecInitType (oap::InitVVectorType::FIRST_VALUE_IS_ONE);
    }

    virtual void TearDown() {
      delete m_arnoldiCuda;
      oap::cuda::Context::Instance().destroy();
    }

    static void multiply (math::Matrix* m_w, math::Matrix* m_v,
                          oap::CuProceduresApi& cuProceduresApi,
                          void* userData, oap::VecMultiplicationType mt)
    {
      UserData* userDataObj = static_cast<UserData*>(userData);

      math::Matrix* hmatrix = userDataObj->hmatrix;
      math::Matrix* hvectorT = userDataObj->hvectorT;
      math::Matrix* dvectorT = userDataObj->dvectorT;
      math::Matrix* dvalue = userDataObj->dvalue;

      for (size_t idx = 0; idx < gRows (hmatrix); ++idx) {
        oap::host::GetTransposeReVector (hvectorT, hmatrix, idx);
        oap::cuda::CopyHostMatrixToDeviceMatrix (dvectorT, hvectorT);
        cuProceduresApi.dotProduct (dvalue, dvectorT, m_v);
        oap::cuda::SetReMatrix (m_w, dvalue, 0, idx);
        PRINT_CUMATRIX(m_w);
        PRINT_CUMATRIX(dvalue);
        PRINT_CUMATRIX(dvectorT);
        PRINT_CUMATRIX(m_v);
      }
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
        oap::HostMatrixPtr hmatrix = oap::host::NewMatrixWithValue (size, size, 0);

        for (size_t xy = 0; xy < size; ++xy) {
          *GetRePtr (hmatrix, xy, xy) = getValue (xy);
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

      for (uint index = 0; index < gColumns (eigenvalues); ++index) {
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

    void runMatrixTest(oap::HostMatrixPtr hmatrix, const std::vector<EigenPair>& eigenPairs, uint hdim, floatt tolerance)
    {
      logInfoLongTest();

      UserData userData = {
              hmatrix,
              oap::host::NewReMatrix(gColumns (hmatrix), 1),
              oap::cuda::NewDeviceReMatrix(gColumns (hmatrix), 1),
              oap::cuda::NewDeviceReMatrix(1, 1)
      };

      CheckUserData checkUserData = {
              &eigenPairs,
              oap::host::NewReMatrix(1, gRows (hmatrix))
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
        revectors.push_back(oap::host::NewReMatrix(1, gRows (hmatrix)));
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

      for (uint index = 0; index < wanted; ++index)
      {
        floatt outcome = m_arnoldiCuda->testOutcome(index);
        logInfo ("Eigenpair = %f", revalues[index]);
        PRINT_MATRIX (revectorsPtr[index]);
        EXPECT_LE (outcome, tolerance);
      }
    }

    void runSmsDataTest(uint index, uint wanted, uint hdim = 32, floatt tolerance = 0) {
      uint columns = data_columns[index];
      uint rows = data_rows[index];

      double* smsmatrix = data_matrices[index];
      double* eigenvalues = data_eigenvalues[index];

      oap::HostMatrixPtr hmatrix = oap::host::NewMatrixCopy<double>(columns, rows, smsmatrix, NULL);

      std::vector<double> ev(eigenvalues, eigenvalues + columns);
      std::vector<EigenPair> eigenPairs;

      for (uint columnIdx = 0; columnIdx < columns; ++columnIdx) {
        eigenPairs.push_back(EigenPair(ev[columnIdx], columnIdx));
      }

      runMatrixTest(hmatrix, getEigenvalues(eigenPairs, wanted), hdim, tolerance);
    }
};

TEST_F(OapArnoldiPackageMatricesTests, Test_1)
{
  m_arnoldiCuda->setQRType (oap::QRType::QRGR);
  runSmsDataTest(Test_CMatrixData1, 2, 6, 0.02);
}

TEST_F(OapArnoldiPackageMatricesTests, Test_2)
{
  m_arnoldiCuda->setQRType (oap::QRType::QRGR);
  runSmsDataTest(Test_CMatrixData2, 3, 6, 0.047);
}

TEST_F(OapArnoldiPackageMatricesTests, Test_3)
{
  m_arnoldiCuda->setQRType (oap::QRType::QRGR);
  runSmsDataTest(Test_CMatrixData3, 6, 12, 0.047);
}

TEST_F(OapArnoldiPackageMatricesTests, Test_4)
{
  m_arnoldiCuda->setQRType (oap::QRType::QRGR);
  runSmsDataTest(Test_CMatrixData4, 6, 12, 0.047);
}
