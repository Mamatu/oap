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


#include <string>
#include <vector>
#include <stdio.h>
#include <pthread.h>
#include "gtest/gtest.h"
#include "oapCudaMatrixUtils.h"
#include "DeviceMatrixKernels.h"
#include "gmock/gmock-generated-function-mockers.h"

typedef std::pair<uintt, uintt> Index;
typedef std::pair<Complex, Index> ValueIndex;
typedef std::vector<ValueIndex> ValueIndexVec;

class OapCudaUtilsTests : public testing::Test {
 public:
  CUresult status;

  oap::cuda::Kernel* m_kernel;

  virtual void SetUp() {
    oap::cuda::Context::Instance().create();
    status = CUDA_SUCCESS;
    m_kernel = new oap::cuda::Kernel();
    m_kernel->load("liboapMatrixCuda.cubin");
  }

  virtual void TearDown() {
    delete m_kernel;
    oap::cuda::Context::Instance().destroy();
  }

  void executeSetGetValueTest(bool isre, bool isim, uintt columns, uintt rows,
                              floatt reexpected, floatt imexpected) {
    ValueIndexVec vec;
    ValueIndex index = ValueIndex(Complex(reexpected, imexpected), Index(0, 0));
    vec.push_back(index);
    executeSetGetValueTest(isre, isim, columns, rows, vec);
  }

  void executeSetGetValueTest(bool isre, bool isim, uintt columns, uintt rows,
                              floatt reexpected) {
    ValueIndexVec vec;
    ValueIndex index = ValueIndex(Complex(reexpected, 0), Index(0, 0));
    vec.push_back(index);
    executeSetGetValueTest(isre, isim, columns, rows, vec);
  }

  void executeSetGetValueTest(bool isre, bool isim, uintt columns, uintt rows,
                              const ValueIndexVec& expecteds) {
    math::Matrix* matrix = oap::cuda::NewDeviceMatrixWithValue (isre, isim, columns, rows, 0.);
    executeSetGetValueTest(matrix, expecteds);
    oap::cuda::DeleteDeviceMatrix(matrix);
  }

  void executeSetGetValueTest (math::Matrix* matrix, const ValueIndexVec& expecteds)
  {
    uintt columns = oap::cuda::GetColumns(matrix);
    for (ValueIndexVec::const_iterator it = expecteds.begin(); it != expecteds.end(); ++it)
    {
      uintt index = it->second.first + columns * it->second.second;
      auto minfo = oap::cuda::GetMatrixInfo (matrix);
      if (minfo.isRe)
      {
        oap::cuda::SetReValue(matrix, index, it->first.re);
        floatt revalue = CudaUtils::GetReValue(matrix, index);
        EXPECT_DOUBLE_EQ(it->first.re, revalue);
      }
      if (minfo.isIm)
      {
        oap::cuda::SetImValue(matrix, index, it->first.im);
        floatt imvalue = CudaUtils::GetImValue(matrix, index);
        EXPECT_DOUBLE_EQ(it->first.im, imvalue);
      }
    }

    for (ValueIndexVec::const_iterator it = expecteds.begin(); it != expecteds.end(); ++it)
    {
      uintt index = it->second.first + columns * it->second.second;
      auto minfo = oap::cuda::GetMatrixInfo (matrix);
      if (minfo.isRe)
      {
        floatt revalue = CudaUtils::GetReValue(matrix, index);
        EXPECT_DOUBLE_EQ(it->first.re, revalue);
      }
      if (minfo.isIm)
      {
        floatt imvalue = CudaUtils::GetImValue(matrix, index);
        EXPECT_DOUBLE_EQ(it->first.im, imvalue);
      }
    }
    CudaUtils::PrintMatrix (matrix);
  }
};

TEST_F(OapCudaUtilsTests, SetGetValueReMatrix) {
  executeSetGetValueTest(true, false, 4, 4, 5.54544f);
}

TEST_F(OapCudaUtilsTests, SetGetValueMatrix) {
  executeSetGetValueTest(true, true, 4, 4, 5.54544f);
}

TEST_F(OapCudaUtilsTests, SetGetValuesMatrix) {
  ValueIndexVec vec;
  vec.push_back(ValueIndex(-2.526556, Index(0, 0)));
  vec.push_back(ValueIndex(0.956565, Index(1, 0)));
  vec.push_back(ValueIndex(-0.956565, Index(0, 1)));
  vec.push_back(ValueIndex(2.526556, Index(1, 1)));
  executeSetGetValueTest(true, true, 4, 4, vec);
}

TEST_F(OapCudaUtilsTests, SetGetValuesMatrix1) {
  ValueIndexVec vec;
  uintt column = 0;
  uintt row = 1;
  floatt s = 0.26726124191242434;
  floatt c = -0.96362411165943151;
  vec.push_back(ValueIndex(-s, Index(column, row)));
  vec.push_back(ValueIndex(c, Index(column, column)));
  vec.push_back(ValueIndex(c, Index(row, row)));
  vec.push_back(ValueIndex(s, Index(row, column)));
  executeSetGetValueTest(true, true, 64, 64, vec);
}

TEST_F(OapCudaUtilsTests, SetGetValuesMatrix2) {
  ValueIndexVec vec;
  uintt column = 0;
  uintt row = 1;
  floatt s = 0.26726124191242434;
  floatt c = -0.96362411165943151;
  vec.push_back(ValueIndex(-s, Index(column, row)));
  vec.push_back(ValueIndex(c, Index(column, column)));
  vec.push_back(ValueIndex(c, Index(row, row)));
  vec.push_back(ValueIndex(s, Index(row, column)));
  math::Matrix* matrix = oap::cuda::NewDeviceMatrix(true, true, 64, 64);
  m_kernel->setDimensionsDevice(matrix);
  DEVICEKernel_SetIdentity(matrix, *m_kernel);
  executeSetGetValueTest(matrix, vec);
  oap::cuda::DeleteDeviceMatrix(matrix);
}

TEST_F(OapCudaUtilsTests, SetGetValuesMatrix3) {
  ValueIndexVec vec;
  uintt column = 5;
  uintt row = 1;
  floatt s = 0.26726124191242434;
  floatt c = -0.96362411165943151;
  vec.push_back(ValueIndex(-s, Index(column, row)));
  vec.push_back(ValueIndex(c, Index(column, column)));
  vec.push_back(ValueIndex(c, Index(row, row)));
  vec.push_back(ValueIndex(s, Index(row, column)));
  math::Matrix* matrix = oap::cuda::NewDeviceMatrix(true, true, 32, 32);
  m_kernel->setDimensionsDevice(matrix);
  DEVICEKernel_SetIdentity(matrix, *m_kernel);
  executeSetGetValueTest(matrix, vec);
  oap::cuda::DeleteDeviceMatrix(matrix);
}
