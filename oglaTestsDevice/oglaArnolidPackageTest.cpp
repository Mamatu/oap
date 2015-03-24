
// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Google Mock - a framework for writing C++ mock classes.
//
// This file tests code in gmock.cc.

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

class OglaArnoldiPackageTests : public testing::Test {
public:

    void EqualsExpectations(floatt* houtput, floatt* doutput, size_t count, floatt bound = 0) {
        for (size_t fa = 0; fa < count; ++fa) {
            Float f1(houtput[fa], bound);
            Float f2(doutput[fa], bound);
            debug("f1 = %f", f1.m_value);
            debug("f2 = %f", f2.m_value);
            EXPECT_TRUE(f1 == f2);
        }
    }

    api::ArnoldiPackage* arnoldiCpu;
    api::ArnoldiPackage* arnoldiCuda;

    virtual void SetUp() {
        arnoldiCpu = new api::ArnoldiPackage(api::ArnoldiPackage::ARNOLDI_CPU);
        arnoldiCuda = new api::ArnoldiPackage(api::ArnoldiPackage::ARNOLDI_GPU);
    }

    virtual void TearDown() {
        delete arnoldiCpu;
        delete arnoldiCuda;
    }
};

TEST_F(OglaArnoldiPackageTests, matrices16x16ev2) {
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

    cuda::Context::Instance().init();

    arnoldiCuda->setMatrix(m);
    arnoldiCuda->setHDimension(h);
    arnoldiCuda->setEigenvaluesBuffer(revs1, imvs1, count);
    arnoldiCuda->start();

    cuda::Context::Instance().destroy();
    EqualsExpectations(revs, revs1, count, 1);
    EqualsExpectations(imvs, imvs1, count, 1);

    host::DeleteMatrix(m);
}

TEST_F(OglaArnoldiPackageTests, matrices64x64ev2) {
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

    cuda::Context::Instance().init();

    arnoldiCuda->setMatrix(m);
    arnoldiCuda->setHDimension(h);
    arnoldiCuda->setEigenvaluesBuffer(revs1, imvs1, count);
    arnoldiCuda->start();

    cuda::Context::Instance().destroy();

    EqualsExpectations(revs, revs1, count, 1);
    EqualsExpectations(imvs, imvs1, count, 1);

    host::DeleteMatrix(m);
}
