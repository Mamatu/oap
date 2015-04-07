
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

#include "MockUtils.h"
#include "oglaCudaStub.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "CuMatrixProcedures/CuMagnitudeUtils.h"
#include "CuMatrixProcedures/CuMagnitudeUtils2.h"
#include <math.h>

const int ct = 32;

class AlgoVersion {
public:

    enum Type {
        VERSION_1 = 1,
        VERSION_2 = 2
    };

private:
    Type m_version;

public:

    AlgoVersion(Type version) : m_version(version) {
        // not implemented
    }

    Type getVersion() const {
        return m_version;
    }

    int getFactor() const {
        return m_version;
    }
};

class OglaMagnitudeTests : public OglaCudaStub {
public:

    virtual void SetUp() {

        OglaCudaStub::SetUp();
    }

    virtual void TearDown() {

        OglaCudaStub::TearDown();
    }
};

class MagnitudeStubImpl : public OglaCudaStub::KernelStub {
public:
    math::Matrix* m_matrix;
    floatt* m_buffer;
    size_t m_bufferLength;

    floatt* m_sums;
    size_t m_sumsLength;

    AlgoVersion m_algoVersion;

    MagnitudeStubImpl(math::Matrix* matrix, uintt columns, uintt rows, AlgoVersion::Type algoVersion) :
        m_algoVersion(algoVersion) {
        m_matrix = matrix;
        calculateDims(columns / m_algoVersion.getFactor(), rows);
        m_bufferLength = blockDim.x * blockDim.y;
        m_sumsLength = gridDim.x * gridDim.y;
        m_buffer = new floatt[m_bufferLength];
        m_sums = new floatt[m_sumsLength];
        memset(m_buffer, 0, sizeof (floatt) * m_bufferLength);
        memset(m_sums, 0, sizeof (floatt) * m_sumsLength);
    }

    virtual ~MagnitudeStubImpl() {
        delete[] m_buffer;
        delete[] m_sums;
    }

    void execute() {
        if (NULL != m_matrix) {
            uintt xlength = GetLength(blockIdx.x, blockDim.x,
                m_matrix->columns / m_algoVersion.getFactor());
            uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
            if (m_algoVersion.getVersion() == AlgoVersion::VERSION_1) {
                cuda_MagnitudeReOpt(m_buffer, m_matrix, sharedIndex, xlength);
            } else if (m_algoVersion.getVersion() == AlgoVersion::VERSION_2) {
                cuda_MagnitudeReOptVer2(m_buffer, m_matrix, sharedIndex, xlength);
            }
        }
    }

    void onChange(OglaCudaStub::KernelStub::ContextChnage contextChange) {
        if (contextChange == OglaCudaStub::KernelStub::CUDA_BLOCK) {
            int actualSum = utils::getSum(m_buffer, m_bufferLength);
            m_sums[gridDim.x * blockIdx.y + blockIdx.x] = actualSum;
            memset(m_buffer, 0, sizeof (floatt) * m_bufferLength);
        }
    }

    uintt getSumPart(uintt index) {
        if (index >= m_sumsLength) {
            return 0;
        }
        return m_sums[index];
    }

    floatt getSum() {
        return sqrt(utils::getSum(m_sums, m_sumsLength));
    }
};

TEST_F(OglaMagnitudeTests, MagnitudeColumns1) {
    floatt hArray[] = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    };

    uint columns = 1;
    uint rows = sizeof (hArray) / sizeof (floatt);

    math::MathOperationsCpu mocpu;

    math::Matrix* matrix = host::NewMatrixCopy(columns, rows, hArray, NULL);

    MagnitudeStubImpl magitudeStubImpl1(matrix, columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&magitudeStubImpl1);

    MagnitudeStubImpl magitudeStubImpl2(matrix, columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&magitudeStubImpl2);

    floatt doutput = magitudeStubImpl1.getSum();
    floatt doutput1 = magitudeStubImpl2.getSum();

    floatt output;
    mocpu.magnitude(&output, matrix);

    host::DeleteMatrix(matrix);
    EXPECT_DOUBLE_EQ(doutput, output);
    EXPECT_DOUBLE_EQ(doutput1, output);
}
