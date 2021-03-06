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


class OapDeviceMatrixUPtrTests : public testing::Test {
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

TEST_F(OapDeviceMatrixUPtrTests, MemLeakUPtrTest)
{
  oap::DeviceMatrixUPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);
}

TEST_F(OapDeviceMatrixUPtrTests, MemLeakUPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr(vec);
}

TEST_F(OapDeviceMatrixUPtrTests, InitializationUPtrsTest)
{
  {
    std::vector<math::ComplexMatrix*> vec = {
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 10),
      oap::cuda::NewDeviceReMatrix(10, 10)
    };

    oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (vec);

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

    oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (vec);

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

    oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (list);

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

    oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (array, 2);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);

    delete[] array;
  }

  {
    math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
    array[0] = oap::cuda::NewDeviceReMatrix(10, 125);
    array[1] = oap::cuda::NewDeviceImMatrix (10, 13);

    oap::DeviceMatricesUPtr ptr (array, 2);

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

    oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (array, 3);

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

    oap::DeviceMatricesUPtr ptr (array, 3);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);
    EXPECT_EQ (array[2], ptr[2]);
  }
}

TEST_F(OapDeviceMatrixUPtrTests, ResetUPtrTest)
{
  oap::DeviceMatrixUPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr.reset (oap::cuda::NewDeviceMatrix(11, 11));
}

TEST_F(OapDeviceMatrixUPtrTests, NotDeallocationTest)
{
  {
    math::ComplexMatrix* rptr = oap::cuda::NewDeviceReMatrix (10, 10);
    {
      oap::DeviceMatrixUPtr ptr (rptr, false); // it will be not deallocated
    }
    oap::cuda::DeleteDeviceMatrix (rptr);
  }
  {
    math::ComplexMatrix* rptr = oap::cuda::NewDeviceReMatrix (10, 10);
    {
      math::ComplexMatrix* rptr1 = oap::cuda::NewDeviceMatrix (10, 10);

      oap::DeviceMatrixUPtr ptr (rptr, false); // it will be not deallocated

      ptr.reset (rptr1); // it will be deallocated
    }
    oap::cuda::DeleteDeviceMatrix (rptr);
  }
  {
    math::ComplexMatrix* rptr = oap::cuda::NewDeviceReMatrix (10, 10);
    math::ComplexMatrix* rptr2 = oap::cuda::NewDeviceImMatrix (100, 100);
    math::ComplexMatrix* rptr3 = oap::cuda::NewDeviceMatrix (100, 101);
    {
      math::ComplexMatrix* rptr1 = oap::cuda::NewDeviceMatrix (100, 10);

      oap::DeviceMatrixUPtr ptr (rptr, false); // it will be not deallocated

      ptr.reset (rptr1); // it will be deallocated
      ptr.reset (rptr2, false); // it will be not deallocated
      ptr.reset (rptr3, true); // it will be deallocated
    }
    oap::cuda::DeleteDeviceMatrix (rptr);
    oap::cuda::DeleteDeviceMatrix (rptr2);
  }
}

TEST_F(OapDeviceMatrixUPtrTests, ResetUPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::vector<math::ComplexMatrix*> vec1 = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  std::initializer_list<math::ComplexMatrix*> list = {
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10),
    oap::cuda::NewDeviceReMatrix(10, 10)
  };

  math::ComplexMatrix** array = new math::ComplexMatrix*[2]; 
  array[0] = oap::cuda::NewDeviceReMatrix(10, 125);
  array[1] = oap::cuda::NewDeviceReMatrix(10, 13);

  math::ComplexMatrix* array1[3] =
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

TEST_F(OapDeviceMatrixUPtrTests, AssignmentUPtrTest)
{
  oap::DeviceMatrixUPtr ptr = oap::cuda::NewDeviceReMatrix (10, 10);

  ptr = oap::cuda::NewDeviceMatrix(11, 11);
  oap::DeviceMatrixUPtr ptr1 = oap::cuda::NewDeviceReMatrix (15, 15);

  ptr = std::move (ptr1);
}

TEST_F(OapDeviceMatrixUPtrTests, AssignmentUPtrsTest)
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
    oap::cuda::NewDeviceMatrix(11, 10)
  };

  oap::DeviceMatricesUPtr ptr = oap::makeDeviceMatricesUPtr (vec);
  ptr = oap::makeDeviceMatricesUPtr (vec1);
}

