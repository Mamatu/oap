
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
#include "MatrixEq.h"
#include "MatrixProcedures.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"

class OglaMatrixCudaTests : public testing::Test {
public:
    math::Matrix* output;
    math::Matrix* eq_output;
    CuMatrix cuMatrix;

    virtual void SetUp() {
        cuda::Context::Instance().init();
        output = NULL;
        eq_output = NULL;
    }

    virtual void TearDown() {
        cuda::Context::Instance().destroy();
        if (output != NULL && eq_output != NULL) {
            bool testPassed = *output == *eq_output;
            if (!testPassed) {
                host::PrintReMatrix("output", output);
                host::PrintReMatrix("eq_output", eq_output);
            }
            EXPECT_TRUE(testPassed);
        }
        host::DeleteMatrix(output);
        host::DeleteMatrix(eq_output);
    }
};

TEST_F(OglaMatrixCudaTests, SetVectorTest) {
    floatt hArray[] = {
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hVArray[] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    };

    output = host::NewReMatrixCopy(10, 10, hArray);
    math::Matrix* V = cuda::NewDeviceMatrix(output, 10, 10);
    math::Matrix* v = cuda::NewDeviceMatrix(output, 1, 10);
    cuda::CopyHostArraysToDeviceMatrix(v, hVArray, NULL);
    cuda::CopyHostMatrixToDeviceMatrix(V, output);

    cuMatrix.setVector(V, 0, v, 10);
    cuda::CopyDeviceMatrixToHostMatrix(output, V);

    cuda::DeleteDeviceMatrix(V);
    cuda::DeleteDeviceMatrix(v);

    eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, SetVectorTest1) {
    floatt hArray[] = {
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hVArray[] = {
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
    };

    output = host::NewReMatrixCopy(10, 10, hArray);
    math::Matrix* V = cuda::NewDeviceMatrix(output, 10, 10);
    math::Matrix* v = cuda::NewDeviceMatrix(output, 2, 10);
    cuda::CopyHostArraysToDeviceMatrix(v, hVArray, NULL);
    cuda::CopyHostMatrixToDeviceMatrix(V, output);

    cuMatrix.setVector(V, 0, v, 10);
    cuda::CopyDeviceMatrixToHostMatrix(output, V);

    cuda::DeleteDeviceMatrix(V);
    cuda::DeleteDeviceMatrix(v);

    eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, GetVectorTest) {
    floatt hArray[] = {
        5,
        5,
        5,
        5,
        5,
        0,
        0,
        0,
        0,
        0,
    };

    floatt hVArray[] = {
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    };

    output = host::NewReMatrixCopy(1, 10, hArray);
    math::Matrix* V = cuda::NewDeviceMatrix(output, 10, 10);
    math::Matrix* v = cuda::NewDeviceMatrix(output, 1, 10);
    cuda::CopyHostArraysToDeviceMatrix(v, hArray, NULL);
    cuda::CopyHostArraysToDeviceMatrix(V, hVArray, NULL);

    cuMatrix.getVector(v, 10, V, 0);
    cuda::CopyDeviceMatrixToHostMatrix(output, v);

    cuda::DeleteDeviceMatrix(V);
    cuda::DeleteDeviceMatrix(v);

    eq_output = host::NewReMatrixCopy(1, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, GetVectorTest1) {
    floatt hArray[] = {
        5, 0,
        5, 0,
        5, 0,
        5, 0,
        5, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };

    floatt hVArray[] = {
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
        1, 1,
    };

    output = host::NewReMatrixCopy(2, 10, hArray);
    math::Matrix* V = cuda::NewDeviceMatrix(output, 10, 10);
    math::Matrix* v = cuda::NewDeviceMatrix(output, 2, 10);
    cuda::CopyHostArraysToDeviceMatrix(v, hArray, NULL);
    cuda::CopyHostArraysToDeviceMatrix(V, hVArray, NULL);

    cuMatrix.getVector(v, 10, V, 0);
    cuda::CopyDeviceMatrixToHostMatrix(output, v);

    cuda::DeleteDeviceMatrix(V);
    cuda::DeleteDeviceMatrix(v);

    eq_output = host::NewReMatrixCopy(2, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, SetIdentityReMatrixTest) {
    floatt hArray[] = {
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    };

    output = host::NewReMatrixCopy(10, 10, hArray);
    math::Matrix* matrix = cuda::NewDeviceMatrix(output, 10, 10);
    cuda::CopyHostArraysToDeviceMatrix(matrix, hArray, NULL);
    cuMatrix.setIdentity(matrix);
    cuda::CopyDeviceMatrixToHostMatrix(output, matrix);

    cuda::DeleteDeviceMatrix(matrix);

    eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, SetDiagonalReMatrixTest) {
    floatt hArray[] = {
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    };

    output = host::NewReMatrixCopy(10, 10, hArray);
    math::Matrix* matrix = cuda::NewDeviceMatrix(output, 10, 10);
    cuda::CopyHostArraysToDeviceMatrix(matrix, hArray, NULL);
    cuMatrix.setDiagonal(matrix, 2, 0);
    cuda::CopyDeviceMatrixToHostMatrix(output, matrix);

    cuda::DeleteDeviceMatrix(matrix);

    eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, MultiplyConstantReMatrixTest) {
    floatt hArray[] = {
        1, 0, 2, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 2, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    };

    floatt hOutputArray[] = {
        5, 0, 10, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 0, 10, 0, 0, 0, 0, 0, 0,
        0, 0, 5, 0, 10, 0, 0, 0, 0, 0,
        0, 0, 0, 5, 0, 10, 0, 0, 0, 0,
        0, 0, 0, 0, 5, 0, 10, 0, 0, 0,
        0, 0, 0, 0, 0, 5, 0, 10, 0, 0,
        0, 0, 0, 0, 0, 0, 5, 0, 10, 0,
        0, 0, 0, 0, 0, 0, 0, 5, 0, 10,
        0, 0, 0, 0, 0, 0, 0, 0, 5, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
    };

    output = host::NewReMatrixCopy(10, 10, hArray);
    math::Matrix* doutput = cuda::NewDeviceMatrix(output, 10, 10);
    math::Matrix* dparam0 = cuda::NewDeviceMatrix(output, 10, 10);
    cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

    cuMatrix.multiplyConstantMatrix(doutput, dparam0, 5);

    cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
    eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, TransponseReMatrixExTest1) {
    floatt hArray[] = {
        1, 1, 1, 0,
        1, 1, 0, 1,
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0,
    };

    floatt hOutputArray[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    output = host::NewReMatrixCopy(10, 4, hArray);
    math::Matrix* doutput = cuda::NewDeviceMatrix(output, 10, 4);
    math::Matrix* dparam0 = cuda::NewDeviceMatrix(output, 4, 10);
    cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

    MatrixEx* matrixEx = cuda::NewDeviceMatrixEx();
    MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
    cuda::SetMatrixEx(matrixEx, &hMatrixEx);

    cuMatrix.transposeMatrixEx(doutput, dparam0, matrixEx);

    cuda::DeleteDeviceMatrixEx(matrixEx);
    cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
    eq_output = host::NewReMatrixCopy(10, 4, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, TransponseReMatrixExTest2) {
    floatt hArray[] = {
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    output = host::NewReMatrixCopy(10, 10, hArray);
    math::Matrix* doutput = cuda::NewDeviceMatrix(output, 10, 10);
    math::Matrix* dparam0 = cuda::NewDeviceMatrix(output, 10, 10);
    cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

    MatrixEx* matrixEx = cuda::NewDeviceMatrixEx();
    MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
    cuda::SetMatrixEx(matrixEx, &hMatrixEx);

    cuMatrix.transposeMatrixEx(doutput, dparam0, matrixEx);

    cuda::DeleteDeviceMatrixEx(matrixEx);
    cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
    eq_output = host::NewReMatrixCopy(10, 10, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, TransponseReMatrixExTest3) {
    floatt hArray[] = {
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hOutputArray[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    output = host::NewReMatrixCopy(10, 4, hArray);
    math::Matrix* doutput = cuda::NewDeviceMatrix(output, 10, 4);
    math::Matrix* dparam0 = cuda::NewDeviceMatrix(output, 10, 10);
    cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);

    MatrixEx* matrixEx = cuda::NewDeviceMatrixEx();
    MatrixEx hMatrixEx = {0, 10, 0, 2, 0, 0};
    cuda::SetMatrixEx(matrixEx, &hMatrixEx);

    cuMatrix.transposeMatrixEx(doutput, dparam0, matrixEx);

    cuda::DeleteDeviceMatrixEx(matrixEx);
    cuda::CopyDeviceMatrixToHostMatrix(output, doutput);
    eq_output = host::NewReMatrixCopy(10, 4, hOutputArray);
}

TEST_F(OglaMatrixCudaTests, MatrixExTest) {
    MatrixEx** dMatrixExs = cuda::NewDeviceMatrixEx(5);
    uintt buffer[] = {
        0, 10, 0, 1, 0, 0,
        0, 1, 0, 15, 0, 20,
        0, 0, 0, 0, 0, 0,
        0, 25, 30, 35, 0, 40,
        0, 1, 0, 2, 3, 5
    };
    cuda::SetMatrixEx(dMatrixExs, buffer, 5);
    MatrixEx matrixEx;

    CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[0], sizeof (MatrixEx));
    EXPECT_TRUE(IsEqual(matrixEx, buffer));

    CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[1], sizeof (MatrixEx));
    EXPECT_TRUE(IsEqual(matrixEx, &buffer[6]));

    CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[2], sizeof (MatrixEx));
    EXPECT_TRUE(IsEqual(matrixEx, &buffer[12]));

    CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[3], sizeof (MatrixEx));
    EXPECT_TRUE(IsEqual(matrixEx, &buffer[18]));

    CudaUtils::CopyDeviceToHost(&matrixEx, dMatrixExs[4], sizeof (MatrixEx));
    EXPECT_TRUE(IsEqual(matrixEx, &buffer[24]));
}

TEST_F(OglaMatrixCudaTests, MagnitudeReMatrixTest) {
    floatt hArray[] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    };

    math::MathOperationsCpu mocpu;

    math::Matrix* matrix = host::NewMatrixCopy(1, 10, hArray, NULL);
    math::Matrix* dparam0 = cuda::NewDeviceMatrix(true, false, 1, 10);
    cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
    floatt doutput;
    cuMatrix.magnitude(doutput, dparam0);
    printf("device_output = %f \n", doutput);

    floatt output;
    mocpu.magnitude(&output, matrix);
    printf("host_output = %f \n", output);
    EXPECT_EQ(output, doutput);
}

TEST_F(OglaMatrixCudaTests, MagnitudeReMatrixTest1) {
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

    math::MathOperationsCpu mocpu;

    math::Matrix* matrix = host::NewMatrixCopy(1, 10, hArray, NULL);
    math::Matrix* dparam0 = cuda::NewDeviceMatrix(true, false, 1, 10);
    cuda::CopyHostArraysToDeviceMatrix(dparam0, hArray, NULL);
    floatt doutput;
    cuMatrix.magnitude(doutput, dparam0);
    printf("device_output = %f \n", doutput);

    floatt output;
    mocpu.magnitude(&output, matrix);
    printf("host_output = %f \n", output);
    EXPECT_EQ(output, doutput);
}

TEST_F(OglaMatrixCudaTests, CompareReMatrixTest1) {
    floatt hArray[] = {
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    math::Matrix* matrix1 = cuda::NewDeviceMatrix(true, false, 10, 10);
    math::Matrix* matrix2 = cuda::NewDeviceMatrix(true, false, 10, 10);
    cuda::CopyHostArraysToDeviceMatrix(matrix1, hArray, NULL);
    cuda::CopyHostArraysToDeviceMatrix(matrix2, hArray, NULL);

    bool result = cuMatrix.compare(matrix1, matrix2);

    cuda::DeleteDeviceMatrix(matrix1);
    cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_TRUE(result);
}

TEST_F(OglaMatrixCudaTests, CompareReMatrixTest1Fail) {
    floatt hArray1[] = {
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    floatt hArray2[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    math::Matrix* matrix1 = cuda::NewDeviceMatrix(true, false, 10, 10);
    math::Matrix* matrix2 = cuda::NewDeviceMatrix(true, false, 10, 10);
    cuda::CopyHostArraysToDeviceMatrix(matrix1, hArray1, NULL);
    cuda::CopyHostArraysToDeviceMatrix(matrix2, hArray2, NULL);

    bool result = cuMatrix.compare(matrix1, matrix2);

    cuda::DeleteDeviceMatrix(matrix1);
    cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_FALSE(result);
}
