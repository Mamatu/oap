/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "KernelExecutor.h"

#include "oapDeviceMatrixPtr.h"
#include "oapDeviceMatrixUPtr.h"


class OapMatrixPtrTests : public testing::Test {
  public:
    virtual void SetUp()
    {
      oap::cuda::Context::Instance().create();
    }

    virtual void TearDown()
    {
      oap::cuda::Context::Instance().destroy();
    }
};

TEST_F(OapMatrixPtrTests, MemLeakPtrTest)
{
  oap::DeviceMatrixPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);
}

TEST_F(OapMatrixPtrTests, MemLeakPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  oap::DeviceMatricesPtr ptr = oap::makeDeviceMatricesPtr(vec);
}

TEST_F(OapMatrixPtrTests, ResetPtrTest)
{
  oap::DeviceMatrixPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr.reset (oap::cuda::NewDeviceMatrix(11, 11));
}

TEST_F(OapMatrixPtrTests, ResetPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::cuda::NewDeviceReMatrix(10, 13),
    oap::cuda::NewDeviceReMatrix(10, 14),
    oap::cuda::NewDeviceReMatrix(10, 15)
  };

  std::initializer_list<math::Matrix*> list = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 100)
  };

  math::Matrix** array =  new math::Matrix*[2];
  array[0] = oap::cuda::NewDeviceReMatrix(10, 125);
  array[1] = oap::cuda::NewDeviceImMatrix (10, 13);

  math::Matrix* array1[3] =
  {
    oap::cuda::NewDeviceReMatrix (110, 25),
    oap::cuda::NewDeviceImMatrix (110, 25),
    oap::cuda::NewDeviceMatrix (110, 25),
  };

  oap::DeviceMatricesPtr ptr = oap::makeDeviceMatricesPtr (vec);
  ptr.reset (vec1);
  ptr.reset (list);
  ptr.reset (array, 2);
  ptr.reset (array1, 3);

  delete[] array;
}

TEST_F(OapMatrixPtrTests, AssignmentPtrTest)
{
  oap::DeviceMatrixPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr = oap::cuda::NewDeviceMatrix(11, 11);

  oap::DeviceMatrixPtr ptr1 = oap::cuda::NewDeviceReMatrix (15, 15);

  ptr = ptr1;
}

TEST_F(OapMatrixPtrTests, AssignmentPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  oap::DeviceMatricesPtr ptr = oap::makeDeviceMatricesPtr (vec);
  ptr = oap::makeDeviceMatricesPtr (vec1);
}

TEST_F(OapMatrixPtrTests, MemLeakUPtrTest)
{
  oap::DeviceMatrixUPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);
}

TEST_F(OapMatrixPtrTests, MemLeakUPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr(vec);
}

TEST_F(OapMatrixPtrTests, ResetUPtrTest)
{
  oap::DeviceMatrixUPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr.reset (oap::cuda::NewDeviceMatrix(11, 11));
}

TEST_F(OapMatrixPtrTests, ResetUPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::initializer_list<math::Matrix*> list = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  math::Matrix** array = new math::Matrix*[2]; 
  array[0] = oap::cuda::NewDeviceReMatrix(10, 125);
  array[1] = oap::cuda::NewDeviceReMatrix(10, 13);

  math::Matrix* array1[3] =
  {
    oap::cuda::NewDeviceReMatrix (110, 25),
    oap::cuda::NewDeviceImMatrix (110, 25),
    oap::cuda::NewDeviceMatrix (110, 25),
  };

  oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (vec);
  ptr.reset (vec1);
  ptr.reset (list);
  ptr.reset (array, 2);
  ptr.reset (array1, 3);

  delete[] array;
}

TEST_F(OapMatrixPtrTests, AssignmentUPtrTest)
{
  oap::DeviceMatrixUPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr = oap::cuda::NewDeviceMatrix(11, 11);
  oap::DeviceMatrixUPtr ptr1 = oap::cuda::NewDeviceReMatrix (15, 15);

  ptr = std::move (ptr1);
}

TEST_F(OapMatrixPtrTests, AssignmentUPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceMatrix(11, 10)
  };

  oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (vec);
  ptr = oap::makeDeviceMatricesUPtr (vec1);
}

