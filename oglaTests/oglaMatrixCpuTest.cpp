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
#include "HostMatrixModules.h"
#include "MathOperationsCpu.h"



class OglaMatrixTests : public testing::Test {
public:
    math::Matrix* matrix1;
    math::Matrix* matrix2;
    math::Matrix* output;
    math::Matrix* eq_output;
    floatt eq_value;
    floatt value;
    int tc;

    virtual void SetUp() {
        tc = 1;
    }

    virtual void TearDown() {
        if (output != NULL && eq_output != NULL) {
            host::PrintReMatrix("output", output);
            host::PrintReMatrix("eq_output", eq_output);
            EXPECT_TRUE(*output == *eq_output);
        } else {
            fprintf(stderr,"value = %f \n", value);
            fprintf(stderr,"eq_value = %f \n", eq_value);
            EXPECT_EQ(eq_value, value);
        }
        host::DeleteMatrix(matrix1);
        host::DeleteMatrix(matrix2);
        host::DeleteMatrix(output);
        host::DeleteMatrix(eq_output);
    }
};

TEST_F(OglaMatrixTests, SubMultiplication1) {
    math::MathOperationsCpu mo;
    floatt outputArray[] = {
        5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    output = host::NewReMatrix(10, 10, 0);
    matrix1 = host::NewReMatrix(10, 10, 1);
    matrix2 = host::NewReMatrix(10, 10, 1);
    eq_output = host::NewReMatrixCopy(10, 10, outputArray);
    mo.setSubColumns(5);
    mo.setSubRows(5);
    mo.setThreadsCount(tc);
    mo.multiply(output, matrix1, matrix2, 5);
}

TEST_F(OglaMatrixTests, Addition) {
    floatt array[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    floatt outputArray[] = {
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 2
    };
    matrix1 = host::NewReMatrixCopy(4, 4, array);
    matrix2 = host::NewReMatrixCopy(4, 4, array);
    eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    output = host::NewMatrix(4, 4);
    math::AdditionOperationCpu additionOperation;
    additionOperation.setThreadsCount(tc);
    additionOperation.setOutputMatrix(output);
    additionOperation.setMatrix1(matrix1);
    additionOperation.setMatrix2(matrix2);
    additionOperation.start();
}

TEST_F(OglaMatrixTests, Substraction) {
    floatt array[] = {1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    floatt outputArray[] = {0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0};
    matrix1 = host::NewReMatrixCopy(4, 4, array);
    matrix2 = host::NewReMatrixCopy(4, 4, array);
    eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    output = host::NewMatrix(4, 4);
    math::SubstracionOperationCpu substractionOperation;
    substractionOperation.setThreadsCount(tc);
    substractionOperation.setOutputMatrix(output);
    substractionOperation.setMatrix1(matrix1);
    substractionOperation.setMatrix2(matrix2);
    substractionOperation.start();
}

TEST_F(OglaMatrixTests, Addition1) {

    floatt array[] = {1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    floatt outputArray[] = {2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0};
    matrix1 = host::NewReMatrixCopy(4, 4, array);
    matrix2 = host::NewReMatrixCopy(4, 4, array);
    output = host::NewMatrix(4, 4);
    eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    math::MathOperationsCpu mo;
    mo.setThreadsCount(tc);
    mo.setSubColumns(2);
    mo.add(output, matrix1, matrix2);
}

TEST_F(OglaMatrixTests, Multiplication) {


    floatt array[] = {
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1
    };

    floatt array1[] = {
        1, 0, 0, 0, 0,
        0, 1, 2, 0, 0,
        0, 0, 1, 5.6, 0,
        0, 3, 0, 1, 0,
        0, 3, 0, 1, 0
    };

    floatt outputArray[] = {
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 3, 0, 0, 0,
        0, 3, 0, 0, 0
    };

    matrix1 = host::NewReMatrixCopy(5, 5, array);
    matrix2 = host::NewReMatrixCopy(5, 5, array1);
    output = host::NewMatrix(5, 5);
    eq_output = host::NewReMatrixCopy(5, 5, outputArray);
    math::DotProductOperationCpu multiplicationOperation;
    multiplicationOperation.setThreadsCount(tc);
    multiplicationOperation.setOutputMatrix(output);
    multiplicationOperation.setSubColumns(2);
    multiplicationOperation.setMatrix1(matrix1);
    multiplicationOperation.setMatrix2(matrix2);
    multiplicationOperation.start();
}

TEST_F(OglaMatrixTests, Diagonalization) {
    floatt array[] = {2, -2, 1,
        -1, 3, -1,
        2, -4, 3};

    floatt array1[] = {2, -1, 1,
        1, 0, -1,
        0, 1, 2};

    floatt array2[] = {1, 2, 0,
        0, 3, 0,
        2, -4, 2};

    floatt array3[] = {-1, 0, -1,
        -1, 0, 0,
        2, 1, 2};

    matrix1 = host::NewReMatrixCopy(3, 3, array2);
    matrix2 = host::NewReMatrixCopy(3, 3, array3);
    output = host::NewMatrix(3, 3);
    eq_output = host::NewMatrix(3, 3);
    math::DiagonalizationOperationCpu diagonalizationOperation;
    diagonalizationOperation.setThreadsCount(tc);
    diagonalizationOperation.setOutputMatrix(output);
    diagonalizationOperation.setMatrix1(matrix1);
    diagonalizationOperation.setMatrix2(matrix2);
    diagonalizationOperation.start();
}

TEST_F(OglaMatrixTests, TensorProduct) {

    floatt array[] = {
        1, 0,
        0, 1
    };
    floatt array1[] = {
        2, 0,
        0, 1
    };
    floatt outputArray[] = {
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    matrix1 = host::NewReMatrixCopy(2, 2, array);
    matrix2 = host::NewReMatrixCopy(2, 2, array1);
    output = host::NewMatrix(4, 4);
    eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    math::TensorProductOperationCpu tpOperation;
    tpOperation.setThreadsCount(tc);
    tpOperation.setOutputMatrix(output);
    tpOperation.setMatrix1(matrix1);
    tpOperation.setMatrix2(matrix2);
    tpOperation.start();
}

TEST_F(OglaMatrixTests, Multiplication1) {
    math::MathOperationsCpu mo;
    HostMatrixUtils funcs;
    floatt outputArray[] = {
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    };
    matrix1 = host::NewReMatrix(10, 10, 1);
    matrix2 = host::NewReMatrix(10, 10, 2);
    output = host::NewReMatrix(10, 10);
    eq_output = host::NewReMatrixCopy(10, 10, outputArray);
    mo.setThreadsCount(tc);
    mo.multiply(output, matrix1, matrix2);
}

TEST_F(OglaMatrixTests, MultiplicationConst) {
    math::MathOperationsCpu mo;
    HostMatrixUtils funcs;
    floatt outputArray[] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    };

    eq_output = host::NewReMatrixCopy(10, 10, outputArray);
    matrix1 = host::NewReMatrix(10, 10, 1);
    matrix2 = NULL;
    floatt m2 = 2.f;
    output = host::NewReMatrix(10, 10);
    mo.setThreadsCount(tc);
    mo.multiply(output, matrix1, &m2);
}

TEST_F(OglaMatrixTests, SubMultiplication) {

    math::MathOperationsCpu mo;
    output = host::NewReMatrix(10, 10, 0);
    eq_output = host::NewReMatrix(10, 10, 0);
    eq_output->reValues[0] = 10;
    matrix1 = host::NewReMatrix(10, 10, 1);
    matrix2 = host::NewReMatrix(10, 10, 1);
    mo.setSubRows(1);
    mo.setSubColumns(1);
    mo.setThreadsCount(tc);
    mo.multiply(output, matrix1, matrix2);
}

TEST_F(OglaMatrixTests, Transpose) {
    math::MathOperationsCpu mo;
    floatt outputArray[] = {
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
        1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000
    };
    output = host::NewReMatrix(5, 10, 0);
    eq_output = host::NewReMatrixCopy(5, 10, outputArray);
    matrix1 = host::NewReMatrix(10, 5, 1);
    matrix2 = NULL;
    mo.setSubRows(4);
    mo.setThreadsCount(tc);
    mo.transpose(output, matrix1);
}