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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "oapHostComplexMatrixPtr.h"
#include "oapHostComplexMatrixUPtr.h"

#include <list>


class OapHostComplexMatrixPtrTests : public testing::Test {
  public:

    virtual void SetUp() {}

    virtual void TearDown() {}
};

TEST_F(OapHostComplexMatrixPtrTests, MemLeakPtrTest)
{
  oap::HostComplexMatrixPtr ptr = oap::host::NewReMatrix (10, 10);
}

TEST_F(OapHostComplexMatrixPtrTests, MemLeakPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  oap::HostComplexMatricesPtr ptr (oap::HostComplexMatricesPtr::make (vec));
}

TEST_F(OapHostComplexMatrixPtrTests, ResetPtrTest)
{
  oap::HostComplexMatrixPtr ptr = oap::host::NewReMatrix (10, 10);

  ptr.reset (oap::host::NewComplexMatrix(11, 11));
}

TEST_F(OapHostComplexMatrixPtrTests, InitializationPtrsTest)
{
  {
    std::vector<math::ComplexMatrix*> vec = {
      oap::host::NewReMatrix(10, 10),
      oap::host::NewReMatrix(10, 10),
      oap::host::NewReMatrix(10, 10)
    };

    oap::HostComplexMatricesPtr ptr (oap::HostComplexMatricesPtr::make (vec));

    for (size_t idx = 0; idx < vec.size(); ++idx)
    {
      EXPECT_EQ (vec[idx], ptr[idx]);
    }
  }

  {
    std::vector<math::ComplexMatrix*> vec = {
      oap::host::NewReMatrix(10, 13),
      oap::host::NewReMatrix(10, 14),
      oap::host::NewReMatrix(10, 15)
    };

    oap::HostComplexMatricesPtr ptr (oap::HostComplexMatricesPtr::make (vec));

    for (size_t idx = 0; idx < vec.size(); ++idx)
    {
      EXPECT_EQ (vec[idx], ptr[idx]);
    }
  }

  {
    std::list<math::ComplexMatrix*> list = {
      oap::host::NewReMatrix(10, 10),
      oap::host::NewReMatrix(10, 10),
      oap::host::NewReMatrix(10, 10),
      oap::host::NewReMatrix(10, 10),
      oap::host::NewReMatrix(10, 100)
    };

    oap::HostComplexMatricesPtr ptr (oap::HostComplexMatricesPtr::make (list));

    size_t idx = 0;
    for (auto it = list.cbegin(); it != list.cend(); ++idx, ++it)
    {
      EXPECT_EQ (*it, ptr[idx]);
    }
  }

  {
    math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
    array[0] = oap::host::NewReMatrix(10, 125);
    array[1] = oap::host::NewImMatrix (10, 13);

    oap::HostComplexMatricesPtr ptr (array, 2, false);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);

    oap::host::DeleteComplexMatrix(array[0]);
    oap::host::DeleteComplexMatrix(array[1]);
    delete[] array;
  }

  {
    math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
    array[0] = oap::host::NewReMatrix(10, 125);
    array[1] = oap::host::NewImMatrix (10, 13);

    oap::HostComplexMatricesPtr ptr (array, 2);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);

    delete[] array;
  }

  {
    math::ComplexMatrix* array [3] =
    {
      oap::host::NewReMatrix(10, 125),
      oap::host::NewImMatrix (10, 13),
      oap::host::NewComplexMatrix (105, 13)
    };

    oap::HostComplexMatricesPtr ptr (array, 3);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);
    EXPECT_EQ (array[2], ptr[2]);
  }

  {
    math::ComplexMatrix* array [3] =
    {
      oap::host::NewReMatrix(10, 125),
      oap::host::NewImMatrix (10, 13),
      oap::host::NewComplexMatrix (105, 13)
    };

    oap::HostComplexMatricesPtr ptr (array, 3);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);
    EXPECT_EQ (array[2], ptr[2]);
  }

}

TEST_F(OapHostComplexMatrixPtrTests, ResetPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  std::vector<math::ComplexMatrix*> vec1 = {
    oap::host::NewReMatrix(10, 13),
    oap::host::NewReMatrix(10, 14),
    oap::host::NewReMatrix(10, 15)
  };

  std::initializer_list<math::ComplexMatrix*> list = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 100)
  };

  math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
  array[0] = oap::host::NewReMatrix(10, 125);
  array[1] = oap::host::NewImMatrix (10, 13);

  math::ComplexMatrix* array1[3] =
  {
    oap::host::NewReMatrix(10, 125),
    oap::host::NewImMatrix (10, 13),
    oap::host::NewComplexMatrix (105, 13)
  };

  oap::HostComplexMatricesPtr ptr (oap::HostComplexMatricesPtr::make (vec));
  ptr.reset (vec1);
  ptr.reset (list);
  ptr.reset (array, 2, false);
  ptr.reset (array1, 3);

  oap::host::DeleteMatrix(array[0]);
  oap::host::DeleteMatrix(array[1]);
  delete[] array;
}

TEST_F(OapHostComplexMatrixPtrTests, AssignmentPtrTest)
{
  oap::HostComplexMatrixPtr ptr = oap::host::NewReMatrix (10, 10);

  ptr = oap::host::NewComplexMatrix(11, 11);

  oap::HostComplexMatrixPtr ptr1 = oap::host::NewReMatrix (15, 15);

  ptr = ptr1;
}

TEST_F(OapHostComplexMatrixPtrTests, AssignmentPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  std::vector<math::ComplexMatrix*> vec1 = {
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10),
    oap::host::NewReMatrix(10, 10)
  };

  oap::HostComplexMatricesPtr ptr (oap::HostComplexMatricesPtr::make (vec));
  ptr = oap::HostComplexMatricesPtr::make (vec1);
}
