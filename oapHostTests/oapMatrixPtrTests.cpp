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


#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "oapHostMatrixPtr.h"
#include "oapHostMatrixUPtr.h"


class OapMatrixPtrTests : public testing::Test {
public:

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(OapMatrixPtrTests, MemLeakPtrTest) {
  std::vector<math::Matrix*> vec = {
    host::NewReMatrix(10, 10),
    host::NewReMatrix(10, 10),
    host::NewReMatrix(10, 10)
  };

  oap::HostMatricesPtr ptr = oap::makeHostMatricesPtr(vec);
}

TEST_F(OapMatrixPtrTests, MemLeakUPtrTest) {
  std::vector<math::Matrix*> vec = {
    host::NewReMatrix(10, 10),
    host::NewReMatrix(10, 10),
    host::NewReMatrix(10, 10)
  };

  oap::HostMatricesUPtr ptr = oap::makeHostMatricesUPtr(vec);
}

