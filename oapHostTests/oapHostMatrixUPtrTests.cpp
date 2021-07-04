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

#include "oapHostComplexMatrixPtr.hpp"
#include "oapHostComplexMatrixUPtr.hpp"

#include <list>


class OapHostComplexMatrixUPtrTests : public testing::Test {
  public:

    virtual void SetUp() {}

    virtual void TearDown() {}
};

TEST_F(OapHostComplexMatrixUPtrTests, MemLeakUPtrTest)
{
  oap::HostComplexMatrixUPtr ptr = oap::chost::NewReMatrix (10, 10);
}

TEST_F(OapHostComplexMatrixUPtrTests, MemLeakUPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10)
  };

  oap::HostComplexMatricesUPtr ptr = (oap::HostComplexMatricesUPtr::make(vec));
}

TEST_F(OapHostComplexMatrixUPtrTests, ResetUPtrTest)
{
  oap::HostComplexMatrixUPtr ptr = oap::chost::NewReMatrix (10, 10);

  ptr.reset (oap::chost::NewComplexMatrix(11, 11));
}

TEST_F(OapHostComplexMatrixUPtrTests, NotDeallocationTest)
{
  {
    math::ComplexMatrix* rptr = oap::chost::NewReMatrix (10, 10);
    {
      oap::HostComplexMatrixUPtr ptr (rptr, false); // it will be not deallocated
    }
    oap::chost::DeleteMatrix (rptr);
  }
  {
    math::ComplexMatrix* rptr = oap::chost::NewReMatrix (10, 10);
    {
      math::ComplexMatrix* rptr1 = oap::chost::NewComplexMatrix (10, 10);

      oap::HostComplexMatrixUPtr ptr (rptr, false); // it will be not deallocated

      ptr.reset (rptr1); // it will be deallocated
    }
    oap::chost::DeleteMatrix (rptr);
  }
  {
    math::ComplexMatrix* rptr = oap::chost::NewReMatrix (10, 10);
    math::ComplexMatrix* rptr2 = oap::chost::NewImMatrix (100, 100);
    math::ComplexMatrix* rptr3 = oap::chost::NewComplexMatrix (100, 101);
    {
      math::ComplexMatrix* rptr1 = oap::chost::NewComplexMatrix (100, 10);

      oap::HostComplexMatrixUPtr ptr (rptr, false); // it will be not deallocated

      ptr.reset (rptr1); // it will be deallocated
      ptr.reset (rptr2, false); // it will be not deallocated
      ptr.reset (rptr3, true); // it will be deallocated
    }
    oap::chost::DeleteMatrix (rptr);
    oap::chost::DeleteMatrix (rptr2);
  }
}

TEST_F(OapHostComplexMatrixUPtrTests, ResetUPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10)
  };

  std::vector<math::ComplexMatrix*> vec1 = {
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10)
  };

  std::initializer_list<math::ComplexMatrix*> list = {
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10)
  };

  math::ComplexMatrix** array = new math::ComplexMatrix*[2]; 
  array[0] = oap::chost::NewReMatrix(10, 125);
  array[1] = oap::chost::NewReMatrix(10, 13);

  math::ComplexMatrix* array1[3] =
  {
    oap::chost::NewReMatrix(10, 125),
    oap::chost::NewImMatrix (10, 13),
    oap::chost::NewComplexMatrix (105, 13)
  };

  oap::HostComplexMatricesUPtr ptr = oap::HostComplexMatricesUPtr::make (vec);
  ptr.reset (vec1);
  ptr.reset (list);
  ptr.reset (array, 2);
  ptr.reset (array1, 3);

  delete[] array;
}

TEST_F(OapHostComplexMatrixUPtrTests, InitializationUPtrsTest)
{
  {
    std::vector<math::ComplexMatrix*> vec = {
      oap::chost::NewReMatrix(10, 10),
      oap::chost::NewReMatrix(10, 10),
      oap::chost::NewReMatrix(10, 10)
    };

    oap::HostComplexMatricesUPtr ptr = oap::HostComplexMatricesUPtr::make (vec);

    for (size_t idx = 0; idx < vec.size(); ++idx)
    {
      EXPECT_EQ (vec[idx], ptr[idx]);
    }
  }

  {
    std::vector<math::ComplexMatrix*> vec = {
      oap::chost::NewReMatrix(10, 13),
      oap::chost::NewReMatrix(10, 14),
      oap::chost::NewReMatrix(10, 15)
    };

    oap::HostComplexMatricesUPtr ptr = oap::HostComplexMatricesUPtr::make (vec);

    for (size_t idx = 0; idx < vec.size(); ++idx)
    {
      EXPECT_EQ (vec[idx], ptr[idx]);
    }
  }

  {
    std::list<math::ComplexMatrix*> list = {
      oap::chost::NewReMatrix(10, 10),
      oap::chost::NewReMatrix(10, 10),
      oap::chost::NewReMatrix(10, 10),
      oap::chost::NewReMatrix(10, 10),
      oap::chost::NewReMatrix(10, 100)
    };

    oap::HostComplexMatricesUPtr ptr = oap::HostComplexMatricesUPtr::make (list);

    size_t idx = 0;
    for (auto it = list.cbegin(); it != list.cend(); ++idx, ++it)
    {
      EXPECT_EQ (*it, ptr[idx]);
    }
  }

  {
    math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
    array[0] = oap::chost::NewReMatrix(10, 125);
    array[1] = oap::chost::NewImMatrix (10, 13);

    oap::HostComplexMatricesUPtr ptr (array, 2, false);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);

    oap::chost::DeleteComplexMatrix(array[0]);
    oap::chost::DeleteComplexMatrix(array[1]);
    delete[] array;
  }

  {
    math::ComplexMatrix** array =  new math::ComplexMatrix*[2];
    array[0] = oap::chost::NewReMatrix(10, 125);
    array[1] = oap::chost::NewImMatrix (10, 13);

    oap::HostComplexMatricesUPtr ptr (array, 2);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);

    delete[] array;
  }

  {
    math::ComplexMatrix* array [3] =
    {
      oap::chost::NewReMatrix(10, 125),
      oap::chost::NewImMatrix (10, 13),
      oap::chost::NewComplexMatrix (105, 13)
    };

    oap::HostComplexMatricesUPtr ptr (array, 3);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);
    EXPECT_EQ (array[2], ptr[2]);
  }

  {
    math::ComplexMatrix* array [3] =
    {
      oap::chost::NewReMatrix(10, 125),
      oap::chost::NewImMatrix (10, 13),
      oap::chost::NewComplexMatrix (105, 13)
    };

    oap::HostComplexMatricesUPtr ptr (array, 3);

    EXPECT_EQ (array[0], ptr[0]);
    EXPECT_EQ (array[1], ptr[1]);
    EXPECT_EQ (array[2], ptr[2]);
  }

}

TEST_F(OapHostComplexMatrixUPtrTests, AssignmentUPtrTest)
{
  oap::HostComplexMatrixUPtr ptr = oap::chost::NewReMatrix (10, 10);

  ptr = oap::chost::NewComplexMatrix(11, 11);
  oap::HostComplexMatrixUPtr ptr1 = oap::chost::NewReMatrix (15, 15);

  ptr = std::move (ptr1);
}

TEST_F(OapHostComplexMatrixUPtrTests, AssignmentUPtrsTest)
{
  std::vector<math::ComplexMatrix*> vec = {
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10)
  };

  std::vector<math::ComplexMatrix*> vec1 = {
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewReMatrix(10, 10),
    oap::chost::NewComplexMatrix(11, 10)
  };

  oap::HostComplexMatricesUPtr ptr = oap::HostComplexMatricesUPtr::make (vec);
  ptr = oap::HostComplexMatricesUPtr::make (vec1);
}

