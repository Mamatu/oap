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
#include "gtest/gtest.h"
#include "ArnoldiMethodProcess.h"
#include "MatricesExamples.h"
#include "KernelExecutor.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"

class Float {
public:

    Float(floatt value, floatt bound = 0) {
        m_value = value;
        m_bound = bound;
    }

    floatt m_value;
    floatt m_bound;

    bool operator==(const Float& value) {
        return (value.m_value - m_bound <= m_value && m_value <= value.m_value + m_bound)
            || (value.m_value - value.m_bound <= m_value && m_value <= value.m_value + value.m_bound);
    }

};

bool operator==(const Float& value1, const Float& value) {
    return (value.m_value - value1.m_bound <= value1.m_value
        && value1.m_value <= value.m_value + value1.m_bound)
        || (value.m_value - value.m_bound <= value1.m_value
        && value1.m_value <= value.m_value + value.m_bound);
}

class OapArnoldiPackageTests : public testing::Test {
public:

    void EqualsExpectations(floatt* houtput, floatt* doutput, size_t count, floatt bound = 0) {
        for (size_t fa = 0; fa < count; ++fa) {
            Float f1(houtput[fa], bound);
            Float f2(doutput[fa], bound);
            EXPECT_TRUE(f1 == f2);
        }
    }

    api::ArnoldiPackage* arnoldiCpu;
    api::ArnoldiPackage* arnoldiCuda;

    virtual void SetUp() {
        arnoldiCpu = new api::ArnoldiPackage(api::ArnoldiPackage::ARNOLDI_CPU);
    }

    virtual void TearDown() {
        delete arnoldiCpu;
    }
};

TEST_F(OapArnoldiPackageTests, matrices16x16ev2) {
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

    host::DeleteMatrix(m);
}

TEST_F(OapArnoldiPackageTests, matrices64x64ev2) {
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

    host::DeleteMatrix(m);
}
