/*
 * Copyright 2016 Marcin Matula
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

#include <string>
#include <stdlib.h>
#include "gtest/gtest.h"
#include "ArnoldiMethodProcess.h"
#include "MatricesExamples.h"
#include "KernelExecutor.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "ArnoldiProcedures.h"

class OapArnoldiPackageTests : public testing::Test {
 public:
  void EqualsExpectations(floatt* houtput, floatt* doutput, size_t count,
                          floatt bound = 0) {
    for (size_t fa = 0; fa < count; ++fa) {
      EXPECT_DOUBLE_EQ(houtput[fa], doutput[fa]);
    }
  }

  api::ArnoldiPackage* arnoldiCpu;
  CuHArnoldiDefault* arnoldiCuda;

  virtual void SetUp() {
    device::Context::Instance().create();
    arnoldiCpu = new api::ArnoldiPackage(api::ArnoldiPackage::ARNOLDI_CPU);
    arnoldiCuda = new CuHArnoldiDefault;
  }

  virtual void TearDown() {
    delete arnoldiCuda;
    delete arnoldiCpu;
    device::Context::Instance().destroy();
  }
};

TEST_F(OapArnoldiPackageTests, DISABLED_matrices16x16ev2) {
  math::Matrix* m = host::NewReMatrixCopy(16, 16, tm16);
  uintt count = 2;
  uintt h = 4;

  floatt revs[] = {0, 0};
  floatt imvs[] = {0, 0};
  floatt revs1[] = {0, 0};
  floatt imvs1[] = {0, 0};

  arnoldiCpu->setMatrix(m);
  arnoldiCpu->setHDimension(h);
  arnoldiCpu->setEigenvaluesBuffer(revs, imvs, count);
  arnoldiCpu->start();

  math::Matrix outputs;

  uintt wanted = 1;

  outputs.reValues = revs1;
  outputs.imValues = imvs1;
  outputs.columns = wanted;

  arnoldiCuda->setRho(1. / 3.14);
  arnoldiCuda->setSortType(ArnUtils::SortLargestReValues);
  arnoldiCuda->setOutputs(&outputs);
  arnoldiCuda->setMatrix(m);
  ArnUtils::MatrixInfo matrixInfo(true, true, 16, 16);
  arnoldiCuda->execute(h, wanted, matrixInfo);

  EqualsExpectations(revs, revs1, count, 1);
  EqualsExpectations(imvs, imvs1, count, 1);

  host::DeleteMatrix(m);
}

TEST_F(OapArnoldiPackageTests, DISABLED_matrices64x64ev2) {
  math::Matrix* m = host::NewReMatrixCopy(64, 64, tm64);
  uintt count = 2;
  uintt h = 8;

  floatt revs[] = {0, 0};
  floatt imvs[] = {0, 0};
  floatt revs1[] = {0, 0};
  floatt imvs1[] = {0, 0};

  arnoldiCpu->setMatrix(m);
  arnoldiCpu->setHDimension(h);
  arnoldiCpu->setEigenvaluesBuffer(revs, imvs, count);
  arnoldiCpu->start();

  math::Matrix outputs;

  uintt wanted = 1;

  outputs.reValues = revs1;
  outputs.imValues = imvs1;
  outputs.columns = wanted;

  arnoldiCuda->setRho(1. / 3.14);
  arnoldiCuda->setSortType(ArnUtils::SortLargestReValues);
  arnoldiCuda->setOutputs(&outputs);
  arnoldiCuda->setMatrix(m);
  ArnUtils::MatrixInfo matrixInfo(true, true, 64, 64);
  arnoldiCuda->execute(h, wanted, matrixInfo);

  EqualsExpectations(revs, revs1, count, 1);
  EqualsExpectations(imvs, imvs1, count, 1);

  host::DeleteMatrix(m);
}
