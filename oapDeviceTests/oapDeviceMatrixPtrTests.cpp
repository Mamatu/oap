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
#include <list>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "KernelExecutor.h"

#include "oapDeviceComplexMatrixPtr.h"
#include "oapDeviceComplexMatrixUPtr.h"


class OapDeviceComplexMatrixPtrTests : public testing::Test {
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

TEST_F(OapDeviceComplexMatrixPtrTests, MemLeakPtrTest)
{
  oap::DeviceComplexMatrixPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);
}

TEST_F(OapDeviceComplexMatrixPtrTests, MemLeakPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr(vec);
}

TEST_F(OapDeviceComplexMatrixPtrTests, InitializationPtrsTest)
{
  {
    std::vector<math::ComplexMatrix*> vec = {
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 10)
    };

    oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr (vec);

    for (size_t idx = 0; idx < vec.size(); ++idx)
    {
      EXPECT_EQ (vec[idx], ptr[idx]);
    }
  }

  {
    std::vector<math::ComplexMatrix*> vec = {
      oap::cuda::NewDeviceReMatrix(10, 13),
      oap::cuda::NewDeviceReMatrix(10, 14),
      oap::cuda::NewDeviceReMatrix(10, 15)
    };

    oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr (vec);

    for (size_t idx = 0; idx < vec.size(); ++idx)
    {
      EXPECT_EQ (vec[idx], ptr[idx]);
    }
  }

  {
    std::list<math::ComplexMatrix*> list = {
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 100)
    };

    oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr (list);

    size_t idx = 0;
    for (auto it = list.cbegin(); it != list.cend(); ++idx, ++it)
    {
      EXPECT_EQ (*it, ptr[idx]);
    }
  }

  {
    math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
    array[0] = oap::cuda::NewDeviceReMatrix(10, 125);
    array[1] = oap::cuda::NewDeviceImMatrix (10, 13);

    oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr (array, 2);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);

    delete[] array;
  }

  {
    math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
    array[0] = oap::cuda::NewDeviceReMatrix(10, 125);
    array[1] = oap::cuda::NewDeviceImMatrix (10, 13);

    oap::DeviceComplexMatricesPtr ptr (array, 2);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);

    delete[] array;
  }

  {
    math::ComplexMatrix* array [3] =
    {
      oap::cuda::NewDeviceReMatrix(10, 125),
      oap::cuda::NewDeviceImMatrix (10, 13),
      oap::cuda::NewDeviceMatrix (105, 13)
    };

    oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr (array, 3);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);
    EXPECT_EQ (array[2], ptr[2]);
  }

  {
    math::ComplexMatrix* array [3] =
    {
      oap::cuda::NewDeviceReMatrix(10, 125),
      oap::cuda::NewDeviceImMatrix (10, 13),
      oap::cuda::NewDeviceMatrix (105, 13)
    };

    oap::DeviceComplexMatricesPtr ptr (array, 3);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);
    EXPECT_EQ (array[2], ptr[2]);
  }
}

TEST_F(OapDeviceComplexMatrixPtrTests, ResetPtrTest)
{
  oap::DeviceComplexMatrixPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr.reset (oap::cuda::NewDeviceMatrix(11, 11));
}

TEST_F(OapDeviceComplexMatrixPtrTests, ResetPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::vector<math::ComplexMatrix*> vec1 = {
    oap::cuda::NewDeviceReMatrix(10, 13),
    oap::cuda::NewDeviceReMatrix(10, 14),
    oap::cuda::NewDeviceReMatrix(10, 15)
  };

  std::initializer_list<math::ComplexMatrix*> list = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 100)
  };

  math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
  array[0] = oap::cuda::NewDeviceReMatrix(10, 125);
  array[1] = oap::cuda::NewDeviceImMatrix (10, 13);

  math::ComplexMatrix* array1[3] =
  {
    oap::cuda::NewDeviceReMatrix (110, 25),
    oap::cuda::NewDeviceImMatrix (110, 25),
    oap::cuda::NewDeviceMatrix (110, 25),
  };

  oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr (vec);
  ptr.reset (vec1);
  ptr.reset (list);
  ptr.reset (array, 2);
  ptr.reset (array1, 3);

  delete[] array;
}

TEST_F(OapDeviceComplexMatrixPtrTests, AssignmentPtrTest)
{
  oap::DeviceComplexMatrixPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr = oap::cuda::NewDeviceMatrix(11, 11);

  oap::DeviceComplexMatrixPtr ptr1 = oap::cuda::NewDeviceReMatrix (15, 15);

  ptr = ptr1;
}

TEST_F(OapDeviceComplexMatrixPtrTests, AssignmentPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::vector<math::ComplexMatrix*> vec1 = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  oap::DeviceComplexMatricesPtr ptr = oap::makeDeviceComplexMatricesPtr (vec);
  ptr = oap::makeDeviceComplexMatricesPtr (vec1);
}
