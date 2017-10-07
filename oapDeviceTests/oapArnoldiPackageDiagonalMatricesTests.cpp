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

class OapArnoldiPackageDiagonalMatricesTests : public testing::Test {
  public:
    CuHArnoldiCallback* arnoldiCuda;
    CuMatrix* cuMatrix;

    virtual void SetUp() {
      device::Context::Instance().create();
      arnoldiCuda = new CuHArnoldiCallback();
      cuMatrix = new CuMatrix();
    }

    virtual void TearDown() {
      delete cuMatrix;
      delete arnoldiCuda;
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

      for (size_t idx = 0; idx < hmatrix->columns; ++idx) {

        host::GetTransposeReVector(hvectorT, hmatrix, idx);
        device::CopyHostMatrixToDeviceMatrix(dvectorT, hvectorT);
        cuMatrix->dotProduct(dvalue, dvectorT, m_w);
        device::SetMatrix(m_w, dvalue, 0, idx);

      }
    }

    oap::HostMatrixPtr createMatrix(size_t size, GetValue getValue)
    {
        oap::HostMatrixPtr hmatrix = host::NewMatrix(size, size, 0);

        for (size_t xy = 0; xy < size; ++xy) {
          hmatrix->reValues[GetIndex(hmatrix, xy, xy)] = getValue(xy);
        }

        return hmatrix;
    }

    struct UserData {
      oap::HostMatrixPtr hmatrix;
      oap::HostMatrixPtr hvectorT;
      oap::DeviceMatrixPtr dvectorT;
      oap::DeviceMatrixPtr dvalue;
      CuMatrix* cuMatrix;
    };

    void executeDiagonalMatrixTest(oap::HostMatrixPtr hmatrix) {
      uintt wanted = 1;
      uint hdim = 32;

      UserData userData = {
              hmatrix,
              host::NewReMatrix(1, hmatrix->rows),
              device::NewDeviceReMatrix(1, hmatrix->rows),
              device::NewDeviceReMatrix(1, 1),
              cuMatrix
      };

      arnoldiCuda->setOutputType(ArnUtils::HOST);

      arnoldiCuda->setCallback(multiply, &hmatrix);
      arnoldiCuda->setBLimit(0.01);
      arnoldiCuda->setRho(1. / 3.14159265359);
      arnoldiCuda->setSortType(ArnUtils::SortLargestReValues);
      arnoldiCuda->setCheckType(ArnUtils::CHECK_FIRST_STOP);
      
      floatt* revalues = new floatt[wanted];
      math::Matrix** revectors = new math::Matrix*[wanted];

      for (size_t idx = 0; idx < wanted; ++idx) {
        revectors[idx] = host::NewReMatrix(1, hmatrix->rows);
      }

      arnoldiCuda->setOutputsEigenvalues(revalues, NULL);
      arnoldiCuda->setOutputsEigenvectors(revectors);
      
      math::MatrixInfo matrixInfo(hmatrix);

      debugLongTest();

      arnoldiCuda->execute(hdim, wanted, matrixInfo);
      delete[] revalues;
    }

    void executeDiagonalMatrixTest(size_t size, GetValue getValue) {
      oap::HostMatrixPtr hmatrix = createMatrix(size, getValue);
      executeDiagonalMatrixTest(hmatrix);
    }
};

TEST_F(OapArnoldiPackageDiagonalMatricesTests, DISABLED_Test1) {
  executeDiagonalMatrixTest(100, [](size_t xy) -> floatt { return xy; });
}
