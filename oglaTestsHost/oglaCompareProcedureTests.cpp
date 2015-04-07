
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
#include "CuMatrixProcedures/CuCompareUtils.h"
#include "CuMatrixProcedures/CuCompareUtils2.h"

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

class OglaCompareTests : public OglaCudaStub {
public:

    virtual void SetUp() {

        OglaCudaStub::SetUp();
    }

    virtual void TearDown() {

        OglaCudaStub::TearDown();
    }

    static int getExpectedResult(uintt columns, uintt rows,
        const Dim3& gridDim, const Dim3& blockIdx, const Dim3& blockDim,
        const AlgoVersion& algoVersion) {

        int factor = algoVersion.getFactor();

        if (gridDim.x == 1 && gridDim.y == 1) {
            return columns * rows;
        }

        uintt xlength = GetLength(blockIdx.x, blockDim.x, columns / factor);
        uintt ylength = GetLength(blockIdx.y, blockDim.y, rows);

        uintt rest = 0;

        if (algoVersion.getVersion() == AlgoVersion::VERSION_2
            && xlength % 2 != 0 && columns % 2 != 0) {
            rest = 3;
            --xlength;
        }

        return (xlength * factor + rest) * ylength;
    }

    static int getExpectedResult(math::Matrix* matrix, const Dim3& gridDim,
        const Dim3& blockIdx, const Dim3& blockDim,
        const AlgoVersion& algoVersion) {
        return getExpectedResult(matrix->columns, matrix->rows, gridDim,
            blockIdx, blockDim, algoVersion);
    }

};

class CompareStubImpl : public OglaCudaStub::KernelStub {
public:
    math::Matrix* m_matrix;
    int* m_buffer;
    size_t m_bufferLength;

    int* m_sums;
    size_t m_sumsLength;

    AlgoVersion m_algoVersion;

    CompareStubImpl(uintt columns, uintt rows, AlgoVersion::Type algoVersion) :
        m_algoVersion(algoVersion) {
        m_matrix = host::NewReMatrix(columns, rows, 0);
        calculateDims(columns / m_algoVersion.getFactor(), rows);
        m_bufferLength = blockDim.x * blockDim.y;
        m_sumsLength = gridDim.x * gridDim.y;
        m_buffer = new int[m_bufferLength];
        m_sums = new int[m_sumsLength];
        memset(m_buffer, 0, sizeof (int) * m_bufferLength);
        memset(m_sums, 0, sizeof (int) * m_sumsLength);
    }

    virtual ~CompareStubImpl() {
        host::DeleteMatrix(m_matrix);
        delete[] m_buffer;
        delete[] m_sums;
    }

    void execute() {
        if (NULL != m_matrix) {
            uintt xlength = GetLength(blockIdx.x, blockDim.x,
                m_matrix->columns / m_algoVersion.getFactor());
            uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
            if (m_algoVersion.getVersion() == AlgoVersion::VERSION_1) {
                cuda_CompareReOpt(m_buffer, m_matrix, m_matrix, sharedIndex, xlength);
            } else if (m_algoVersion.getVersion() == AlgoVersion::VERSION_2) {
                cuda_CompareReOptVer2(m_buffer, m_matrix, m_matrix, sharedIndex, xlength);
            }
        }
    }

    void onChange(OglaCudaStub::KernelStub::ContextChnage contextChange) {
        if (contextChange == OglaCudaStub::KernelStub::CUDA_BLOCK) {
            int actualSum = utils::getSum(m_buffer, m_bufferLength);
            m_sums[gridDim.x * blockIdx.y + blockIdx.x] = actualSum;
            int expectedSum = OglaCompareTests::getExpectedResult(m_matrix,
                gridDim, blockIdx, blockDim, m_algoVersion);
            EXPECT_THAT(actualSum, IsEqualSum(expectedSum, m_buffer, m_bufferLength,
                utils::cudaDimsToStr()));
            memset(m_buffer, 0, sizeof (int) * m_bufferLength);
        }
    }

    uintt getSumPart(uintt index) {
        if (index >= m_sumsLength) {
            return 0;
        }
        return m_sums[index];
    }

    uintt getSum() {
        return utils::getSum(m_sums, m_sumsLength);
    }
};

TEST_F(OglaCompareTests, CoverTestTestAlgoVer1) {
    uintt columns = 64;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    EXPECT_THAT(compareStubImpl.m_matrix, MatrixValuesAreEqual(0));
}

TEST_F(OglaCompareTests, CompareReMatrixOneBlockCoverTestAlgoVer1) {
    uintt columns = 32;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixFixedSizeCoverTestAlgoVer1) {
    uintt columns = 64;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixCoverTestAlgoVer1) {
    uint columns = 50;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixCoverBigDataTestAlgoVer1) {
    uint columns = 90;
    uintt rows = 50;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigDataAlgoVer1) {

    uintt columns = 50;
    uintt rows = 32;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData1AlgoVer1) {

    uintt columns = 50;
    uintt rows = 50;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData2AlgoVer1) {

    uintt columns = 70;
    uintt rows = 70;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData3AlgoVer1) {

    uintt columns = 111;
    uintt rows = 111;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData3LAlgoVer1) {

    uintt columns = 11;
    uintt rows = 11;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData4AlgoVer1) {

    uintt columns = 1000;
    uintt rows = 1000;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_1);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CoverTestTestAlgoVer2) {
    uintt columns = 64;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    EXPECT_THAT(compareStubImpl.m_matrix, MatrixValuesAreEqual(0));
}

TEST_F(OglaCompareTests, CompareReMatrixOneBlockCoverTestAlgoVer2) {
    uintt columns = 32;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixFixedSizeCoverTestAlgoVer2) {
    uintt columns = 64;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixCoverTestAlgoVer2) {
    uint columns = 50;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixCoverBigDataTestAlgoVer2) {
    uint columns = 90;
    uintt rows = 50;
    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigDataAlgoVer2) {

    uintt columns = 50;
    uintt rows = 32;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData1AlgoVer2) {

    uintt columns = 50;
    uintt rows = 50;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData2AlgoVer2) {

    uintt columns = 70;
    uintt rows = 70;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData3AlgoVer2) {

    uintt columns = 111;
    uintt rows = 111;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData3LAlgoVer2) {

    uintt columns = 11;
    uintt rows = 11;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCompareTests, CompareReMatrixTestBigData4AlgoVer2) {

    uintt columns = 1000;
    uintt rows = 1000;

    CompareStubImpl compareStubImpl(columns, rows, AlgoVersion::VERSION_2);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}