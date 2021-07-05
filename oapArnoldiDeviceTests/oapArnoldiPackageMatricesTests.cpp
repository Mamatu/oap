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

#include <algorithm>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "gtest/gtest.h"

#include "ArnoldiProceduresImpl.hpp"
#include "MatchersUtils.hpp"

#include "Config.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "KernelExecutor.hpp"
#include "oapHostComplexMatrixApi.hpp"
#include "MatchersUtils.hpp"
#include "oapEigen.hpp"

#include "oapDeviceComplexMatrixPtr.hpp"
#include "oapHostComplexMatrixPtr.hpp"

#include "CMatrixDataCollector.hpp"
#include "InfoType.hpp"

using GetValue = std::function<floatt(size_t xy)>;
using HostComplexMatrixPtrs = std::vector<oap::HostComplexMatrixPtr>;

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

    static void multiply (math::ComplexMatrix* m_w, math::ComplexMatrix* m_v,
                          oap::CuProceduresApi& cuProceduresApi,
                          void* userData, oap::VecMultiplicationType mt)
    {
      UserData* userDataObj = static_cast<UserData*>(userData);

      math::ComplexMatrix* hmatrix = userDataObj->hmatrix;
      math::ComplexMatrix* hvectorT = userDataObj->hvectorT;
      math::ComplexMatrix* dvectorT = userDataObj->dvectorT;
      math::ComplexMatrix* dvalue = userDataObj->dvalue;

      for (size_t idx = 0; idx < gRows (hmatrix); ++idx) {
        oap::chost::GetTransposeReVector (hvectorT, hmatrix, idx);
        oap::cuda::CopyHostMatrixToDeviceMatrix (dvectorT, hvectorT);
        cuProceduresApi.dotProduct (dvalue, dvectorT, m_v);
        oap::cuda::SetReMatrix (m_w, dvalue, 0, idx);
        //PRINT_CUMATRIX(m_w);
        //PRINT_CUMATRIX(dvalue);
        //PRINT_CUMATRIX(dvectorT);
        //PRINT_CUMATRIX(m_v);
      }
    }

    static bool check(floatt reevalue, floatt imevalue, math::ComplexMatrix* vector, uint index, uint max, void* userData) {
      CheckUserData* checkUserData = static_cast<CheckUserData*>(userData);

      oap::HostComplexMatrixPtr eigenvectors = checkUserData->eigenvectors;
      oap::HostComplexMatrixPtr tmpVector = checkUserData->tmpVector;

      const std::vector<EigenPair>* eigenPairs = checkUserData->eigenPairs;

      EigenPair eigenPair = (*eigenPairs)[index];
      uint vectorIndex = eigenPair.getIndex();

      oap::chost::GetVector(tmpVector, eigenvectors, vectorIndex);

      return true;
    }

    oap::HostComplexMatrixPtr createSquareMatrix(size_t size, GetValue getValue)
    {
        oap::HostComplexMatrixPtr hmatrix = oap::chost::NewComplexMatrixWithValue (size, size, 0);

        for (size_t xy = 0; xy < size; ++xy) {
          *GetRePtr (hmatrix, xy, xy) = getValue (xy);
        }

        return hmatrix;
    }

    oap::HostComplexMatrixPtr loadSMSMatrix(const std::string& dir) {
      return oap::chost::ReadMatrix(dir + "/smsmatrix.matrix");
    }

    oap::HostComplexMatrixPtr loadEigenvaluesMatrix(const std::string& dir) {
      return oap::chost::ReadMatrix(dir + "/eigenvalues.matrix");
    }

    std::vector<EigenPair> getEigenvalues(const std::vector<EigenPair>& avalues, uint wanted) {

      std::vector<EigenPair> values(avalues);

      std::sort(values.begin(), values.end(), ArnUtils::SortLargestReValues);

      std::vector<EigenPair> output(values.begin(), values.begin() + wanted);

      return output;
    }

    oap::HostComplexMatrixPtr loadEigenvector(const std::string& dir, uint index) {
      std::string filename = "eigenvector.matrix";
      filename += std::to_string(index);
      return oap::chost::ReadMatrix(dir + "/" + filename);
    }

    HostComplexMatrixPtrs loadMatrices(const std::string& dir) {
      HostComplexMatrixPtrs ptrs;

      oap::HostComplexMatrixPtr smsMatrix = loadSMSMatrix(dir);
      oap::HostComplexMatrixPtr eigenvalues = loadEigenvaluesMatrix(dir);

      ptrs.push_back(smsMatrix);
      ptrs.push_back(eigenvalues);

      for (uint index = 0; index < gColumns (eigenvalues); ++index) {
        ptrs.push_back(loadEigenvector(dir, index));     
      }

      return ptrs;
    }

    struct UserData {
      oap::HostComplexMatrixPtr hmatrix;
      oap::HostComplexMatrixPtr hvectorT;
      oap::DeviceComplexMatrixPtr dvectorT;
      oap::DeviceComplexMatrixPtr dvalue;
    };

    struct CheckUserData {
      const std::vector<EigenPair>* eigenPairs;
      oap::HostComplexMatrixPtr eigenvectors;
      oap::HostComplexMatrixPtr tmpVector;
    };

    void runMatrixTest(oap::HostComplexMatrixPtr hmatrix, const std::vector<EigenPair>& eigenPairs, uint hdim, floatt tolerance)
    {
      logInfoLongTest();

      UserData userData = {
              hmatrix,
              oap::chost::NewReMatrix(gColumns (hmatrix), 1),
              oap::cuda::NewDeviceReMatrix(gColumns (hmatrix), 1),
              oap::cuda::NewDeviceReMatrix(1, 1)
      };

      CheckUserData checkUserData = {
              &eigenPairs,
              oap::chost::NewReMatrix(1, gRows (hmatrix))
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

      std::vector<math::ComplexMatrix*> revectors;

      for (size_t idx = 0; idx < wanted; ++idx) {
        revectors.push_back(oap::chost::NewReMatrix(1, gRows (hmatrix)));
      }

      oap::HostComplexMatricesPtr revectorsPtr = oap::HostComplexMatricesPtr::make (revectors);

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
        EXPECT_LE (outcome, tolerance);
      }
    }

    void runSmsDataTest(uint index, uint wanted, uint hdim = 32, floatt tolerance = 0) {
      uint columns = data_columns[index];
      uint rows = data_rows[index];

      double* smsmatrix = data_matrices[index];
      double* eigenvalues = data_eigenvalues[index];

      oap::HostComplexMatrixPtr hmatrix = oap::chost::NewComplexMatrixCopy<double>(columns, rows, smsmatrix, NULL);

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
