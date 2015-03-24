
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

const int ct = 32;

class OglaCoverTests : public OglaCudaStub {
public:

    virtual void SetUp() {

        OglaCudaStub::SetUp();
    }

    virtual void TearDown() {

        OglaCudaStub::TearDown();
    }

    static int getExpectedResult(uintt columns, uintt rows,
        const Dim3& gridDim, const Dim3& blockIdx, const Dim3& blockDim) {

        int factor = 1;

        if (gridDim.x == 1 && gridDim.y == 1) {
            return columns * rows * factor;
        }

        uintt rdimx = columns % blockDim.x;
        uintt rdimy = rows % blockDim.y;
        if (rdimx == 0) {
            rdimx = columns / blockDim.x;
            if (rdimx < 32) {
                rdimx = blockDim.x;
            }
        }
        
        if (rdimy == 0) {
            rdimy = rows / blockDim.y;
            if (rdimy < 32) {
                rdimy = blockDim.y;
            }
        }

        if (blockIdx.x < gridDim.x - 1 && blockIdx.y < gridDim.y - 1) {
            return blockDim.x * blockDim.y * factor;
        } else if (blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
            return rdimx * rdimy * factor;
        } else if (blockIdx.x == gridDim.x - 1) {
            return blockDim.y * rdimx * factor;
        } else if (blockIdx.y == gridDim.y - 1) {
            return blockDim.x * rdimy * factor;
        }
    }

    static int getExpectedResult(math::Matrix* matrix,
        const Dim3& gridDim, const Dim3& blockIdx, const Dim3& blockDim) {
        return getExpectedResult(matrix->columns, matrix->rows, gridDim, blockIdx, blockDim);
    }

};

class CompareStubImpl : public OglaCudaStub::KernelStub {
public:
    math::Matrix* m_matrix;
    int* m_buffer;
    size_t m_bufferLength;

    int* m_sums;
    size_t m_sumsLength;

    CompareStubImpl(uintt columns, uintt rows) {
        m_matrix = host::NewReMatrix(columns, rows, 0);
        calculateDims(columns, rows);
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
            uintt xlength = GetLength(blockIdx.x, blockDim.x, m_matrix->columns);
            uintt sharedIndex = threadIdx.y * xlength + threadIdx.x;
            cuda_CompareReOpt(m_buffer, m_matrix, m_matrix, sharedIndex, xlength);
        }
    }

    void onChange(OglaCudaStub::KernelStub::ContextChnage contextChange) {
        if (contextChange == OglaCudaStub::KernelStub::CUDA_BLOCK) {
            int actualSum = utils::getSum(m_buffer, m_bufferLength);
            m_sums[gridDim.x * blockIdx.y + blockIdx.x] = actualSum;
            int expectedSum = OglaCoverTests::getExpectedResult(m_matrix,
                gridDim, blockIdx, blockDim);
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

TEST_F(OglaCoverTests, CoverTestTest) {
    uintt columns = 64;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows);
    EXPECT_THAT(compareStubImpl.m_matrix, MatrixValuesAreEqual(0));
}

TEST_F(OglaCoverTests, CompareReMatrixOneBlockCoverTest) {
    uintt columns = 32;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixFixedSizeCoverTest) {
    uintt columns = 64;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixCoverTest) {
    uint columns = 50;
    uintt rows = 32;
    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixCoverBigDataTest) {
    uint columns = 90;
    uintt rows = 50;
    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);
    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();
    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixTestBigData) {

    uintt columns = 50;
    uintt rows = 32;

    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixTestBigData1) {

    uintt columns = 50;
    uintt rows = 50;

    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixTestBigData2) {

    uintt columns = 70;
    uintt rows = 70;

    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixTestBigData3) {

    uintt columns = 111;
    uintt rows = 111;

    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}

TEST_F(OglaCoverTests, CompareReMatrixTestBigData4) {

    uintt columns = 1000;
    uintt rows = 1000;

    CompareStubImpl compareStubImpl(columns, rows);
    executeKernelSync(&compareStubImpl);

    uintt expected = columns * rows;
    uintt sum = compareStubImpl.getSum();

    EXPECT_EQ(expected, sum);
}