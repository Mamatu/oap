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

#include "Config.h"
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"
#include "HostMatrixUtils.h"
#include "MatchersUtils.h"
#include "MathOperationsCpu.h"

#include "oapDeviceMatrixPtr.h"
#include "oapHostMatrixPtr.h"

using GetValue = std::function<floatt(size_t xy)>;
using HostMatrixPtrs = std::vector<oap::HostMatrixPtr>;

class OapArnoldiPackageMatricesTests : public testing::Test {
  public:
    CuHArnoldiCallback* m_arnoldiCuda;
    CuMatrix* m_cuMatrix;

    virtual void SetUp() {
      device::Context::Instance().create();
      m_arnoldiCuda = new CuHArnoldiCallback();
      m_cuMatrix = new CuMatrix();
    }

    virtual void TearDown() {
      delete m_cuMatrix;
      delete m_arnoldiCuda;
      device::Context::Instance().destroy();
    }

    static void multiply(math::Matrix* m_w, math::Matrix* m_v, 
        void* userData, CuHArnoldi::MultiplicationType mt)
    {
      UserData* userDataObj = static_cast<UserData*>(userData);

      math::Matrix* hmatrix = userDataObj->hmatrix;
      math::Matrix* hvectorT = userDataObj->hvectorT;
      math::Matrix* dvectorT = userDataObj->dvectorT;
      math::Matrix* dvalue = userDataObj->dvalue;
      CuMatrix* cuMatrix = userDataObj->cuMatrix;

      for (size_t idx = 0; idx < hmatrix->rows; ++idx) {
        host::GetTransposeReVector(hvectorT, hmatrix, idx);
        device::CopyHostMatrixToDeviceMatrix(dvectorT, hvectorT);
        cuMatrix->dotProduct(dvalue, dvectorT, m_v);
        device::SetReMatrix(m_w, dvalue, 0, idx);
      }

      device::PrintMatrix("m_w =", m_w);
      device::PrintMatrix("m_v =", m_v);
    }

    oap::HostMatrixPtr createSquareMatrix(size_t size, GetValue getValue)
    {
        oap::HostMatrixPtr hmatrix = host::NewMatrix(size, size, 0);

        for (size_t xy = 0; xy < size; ++xy) {
          hmatrix->reValues[GetIndex(hmatrix, xy, xy)] = getValue(xy);
        }

        return hmatrix;
    }

    oap::HostMatrixPtr loadSMSMatrix(const std::string& dir) {
      return host::ReadMatrix(dir + "/smsmatrix.matrix");
    }

    oap::HostMatrixPtr loadEigenvaluesMatrix(const std::string& dir) {
      return host::ReadMatrix(dir + "/eigenvalues.matrix");
    }

    std::vector<floatt> getEigenvalues(oap::HostMatrixPtr eMatrix) {

      std::vector<floatt> values;

      for (uintt index = 0; index < eMatrix->columns; ++index) {
        values.push_back(eMatrix->reValues[index + eMatrix->columns * index]);
      }

      std::sort(values.begin(), values.end(), std::greater<floatt>());
      return values;
    }

    oap::HostMatrixPtr loadEigenvector(const std::string& dir, uint index) {
      std::string filename = "eigenvector.matrix";
      filename += std::to_string(index);
      return host::ReadMatrix(dir + "/" + filename);
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
      CuMatrix* cuMatrix;
    };

    void executeMatrixTest(oap::HostMatrixPtr hmatrix, const std::vector<floatt>& expectedValues) {
      const uint hdim = 32;

      debugLongTest();

      host::PrintMatrix("h =", hmatrix);

      UserData userData = {
              hmatrix,
              host::NewReMatrix(1, hmatrix->rows),
              device::NewDeviceReMatrix(1, hmatrix->rows),
              device::NewDeviceReMatrix(1, 1),
              m_cuMatrix
      };

      m_arnoldiCuda->setOutputType(ArnUtils::HOST);

      m_arnoldiCuda->setCallback(multiply, &userData);
      m_arnoldiCuda->setBLimit(0.01);
      m_arnoldiCuda->setRho(1. / 3.14159265359);
      m_arnoldiCuda->setSortType(ArnUtils::SortLargestReValues);
      //m_arnoldiCuda->setCheckType(ArnUtils::CHECK_FIRST_STOP);

      uint wanted = expectedValues.size();

      std::unique_ptr<floatt[]> revalues(new floatt[wanted]);

      std::vector<math::Matrix*> revectors;

      for (size_t idx = 0; idx < wanted; ++idx) {
        revectors.push_back(host::NewReMatrix(1, hmatrix->rows));
      }

      oap::HostMatricesPtr revectorsPtr = oap::makeHostMatricesPtr(revectors);

      m_arnoldiCuda->setOutputsEigenvalues(revalues.get(), NULL);
      m_arnoldiCuda->setOutputsEigenvectors(revectorsPtr);
      
      math::MatrixInfo matrixInfo(hmatrix);

      m_arnoldiCuda->execute(hdim, wanted, matrixInfo);

      std::vector<floatt> outputValues(&revalues[0], &revalues[wanted]);

      EXPECT_EQ(expectedValues, outputValues);
    }

    void executeMatrixTest(size_t size, GetValue getValue, const std::vector<floatt>& expectedValues) {
      oap::HostMatrixPtr hmatrix = createSquareMatrix(size, getValue);
      executeMatrixTest(hmatrix, expectedValues);
    }

    void executeMatrixTest(const std::string& dirName) {
      HostMatrixPtrs ptrs = loadMatrices(utils::Config::getPathInOap("oapDeviceTests/data/smsdata/" + dirName));
      std::vector<floatt> evalues = getEigenvalues(ptrs[1]);
      executeMatrixTest(ptrs[0], evalues);
    }
};

TEST_F(OapArnoldiPackageMatricesTests, DISABLE_SMSTest1) {
  executeMatrixTest("smsdata1");
}
