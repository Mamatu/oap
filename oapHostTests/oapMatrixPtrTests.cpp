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

#include "oapHostMatrixPtr.h"
#include "oapHostMatrixUPtr.h"


class OapMatrixPtrTests : public testing::Test {
  public:

    virtual void SetUp() {}

    virtual void TearDown() {}
};

TEST_F(OapMatrixPtrTests, MemLeakPtrTest)
{
  oap::HostMatrixPtr ptr = oap::host::NewReMatrix (10, 10);
}

TEST_F(OapMatrixPtrTests, MemLeakPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  oap::HostMatricesPtr ptr = oap::makeHostMatricesPtr(vec);
}

TEST_F(OapMatrixPtrTests, ResetPtrTest)
{
  oap::HostMatrixPtr ptr = oap::host::NewReMatrix (10, 10);

  ptr.reset (oap::host::NewMatrix(11, 11));
}

TEST_F(OapMatrixPtrTests, ResetPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::host::NewReMatrix(10, 13),
    oap::host::NewReMatrix(10, 14),
    oap::host::NewReMatrix(10, 15)
  };

  std::initializer_list<math::Matrix*> list = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 100)
  };

  math::Matrix** array =  new math::Matrix*[2];
  array[0] = oap::host::NewReMatrix(10, 125);
  array[1] = oap::host::NewImMatrix (10, 13);

  math::Matrix* array1[3] =
  {
    oap::host::NewReMatrix(10, 125),
    oap::host::NewImMatrix (10, 13),
    oap::host::NewMatrix (105, 13)
  };

  oap::HostMatricesPtr ptr = oap::makeHostMatricesPtr (vec);
  ptr.reset (vec1);
  ptr.reset (list);
  ptr.reset (array, 2);
  ptr.reset (array1, 3);

  delete[] array;
}

TEST_F(OapMatrixPtrTests, AssignmentPtrTest)
{
  oap::HostMatrixPtr ptr = oap::host::NewReMatrix (10, 10);

  ptr = oap::host::NewMatrix(11, 11);

  oap::HostMatrixPtr ptr1 = oap::host::NewReMatrix (15, 15);

  ptr = ptr1;
}

TEST_F(OapMatrixPtrTests, AssignmentPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  oap::HostMatricesPtr ptr = oap::makeHostMatricesPtr (vec);
  ptr = oap::makeHostMatricesPtr (vec1);
}

TEST_F(OapMatrixPtrTests, MemLeakUPtrTest)
{
  oap::HostMatrixUPtr ptr = oap::host::NewReMatrix (10, 10);
}

TEST_F(OapMatrixPtrTests, MemLeakUPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  oap::HostMatricesUPtr ptr = oap::makeHostMatricesUPtr(vec);
}

TEST_F(OapMatrixPtrTests, ResetUPtrTest)
{
  oap::HostMatrixUPtr ptr = oap::host::NewReMatrix (10, 10);

  ptr.reset (oap::host::NewMatrix(11, 11));
}

TEST_F(OapMatrixPtrTests, ResetUPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  std::initializer_list<math::Matrix*> list = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  math::Matrix** array = new math::Matrix*[2]; 
  array[0] = oap::host::NewReMatrix(10, 125);
  array[1] = oap::host::NewReMatrix(10, 13);

  math::Matrix* array1[3] =
  {
    oap::host::NewReMatrix(10, 125),
    oap::host::NewImMatrix (10, 13),
    oap::host::NewMatrix (105, 13)
  };

  oap::HostMatricesUPtr ptr = oap::makeHostMatricesUPtr (vec);
  ptr.reset (vec1);
  ptr.reset (list);
  ptr.reset (array, 2);
  ptr.reset (array1, 3);

  delete[] array;
}

TEST_F(OapMatrixPtrTests, AssignmentUPtrTest)
{
  oap::HostMatrixUPtr ptr = oap::host::NewReMatrix (10, 10);

  ptr = oap::host::NewMatrix(11, 11);
  oap::HostMatrixUPtr ptr1 = oap::host::NewReMatrix (15, 15);

  ptr = std::move (ptr1);
}

TEST_F(OapMatrixPtrTests, AssignmentUPtrsTest)
{
  std::vector<math::Matrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  std::vector<math::Matrix*> vec1 = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewMatrix(11, 10)
  };

  oap::HostMatricesUPtr ptr = oap::makeHostMatricesUPtr (vec);
  ptr = oap::makeHostMatricesUPtr (vec1);
}

