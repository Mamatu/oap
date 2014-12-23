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

TEST_F(OglaMatrixTests, Determinant) {

    math::MathOperationsCpu mo;
    floatt array[] = {
        -2, 2, -3,
        -1, 1, 3,
        2, 0, -1
    };
    output = NULL;
    eq_output = NULL;
    matrix1 = host::NewReMatrixCopy(3, 3, array);
    matrix2 = NULL;
    mo.setThreadsCount(tc);
    eq_value = 0;
    mo.det(&value, matrix1);
}

TEST_F(OglaMatrixTests, Determinant2) {

    math::MathOperationsCpu mo;
    floatt array[] = {
        -5618.170, 5543.450, -1.932, 150.394,
        5545.490, -10936.100, 5615.610, 152.460,
        0.000, 5615.550, -5772.430, 220.557,
        0.000, 0.000, 72.205, -11234.900
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    mo.setThreadsCount(tc);
    mo.det(&value, o);
    host::DeleteMatrix(o);

}

TEST_F(OglaMatrixTests, Determinant3) {

    math::MathOperationsCpu mo;
    floatt array[] = {
        -5618.170, 5543.450, -1.932, 150.394,
        5545.490, -10936.100, 5615.610, 152.460,
        0.000, 5615.550, -5772.430, 220.557,
        0.000, 0.000, 72.205, -11234.900
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    floatt value;
    mo.setThreadsCount(tc);
    mo.det(&value, o);
    fprintf(stderr, "value == %f \n", value);
    host::DeleteMatrix(o);

}

TEST_F(OglaMatrixTests, Determinant1) {

    math::MathOperationsCpu mo;
    floatt array[] = {
        3, -5, 3,
        2, 1, -1,
        1, 0, 4
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    floatt value;
    mo.setThreadsCount(tc);
    mo.det(&value, o);
    fprintf(stderr, "value == %f \n", value);
    host::DeleteMatrix(o);

}

TEST_F(OglaMatrixTests, Cut) {

    math::MathOperationsCpu mo;
    floatt array[] = {
        -2, 2, -3,
        -1, 1, 3,
        2, 0, -1
    };
    math::Matrix* o = host::NewReMatrixCopy(3, 3, array);
    math::Matrix* o1 = host::NewReMatrix(2, 2, 0);
    host::Copy(o1, o, 0, 2);
    host::PrintReMatrix(o1);
    host::DeleteMatrix(o);
    host::DeleteMatrix(o1);

}

TEST_F(OglaMatrixTests, SubMultiplication2) {
    floatt w[] = {12, 0, 0};
    floatt A[] = {12, -51, 4,
        6, 167, -68,
        -4, 24, -41};
    floatt v[] = {1, 0, 0};
}

TEST_F(OglaMatrixTests, DotProduct) {
    floatt Aa[] = {11013.7, 5543.45, -1.93192, 150.394,
        5545.49, 5695.82, 5615.61, 152.46,
        0, 5615.55, 10859.5, 220.557,
        0, 0, 72.2053, 5396.97};
    floatt tqa[] = {0.893171, 0.449718, 0, 0,
        -0.188611, 0.374595, 0.907802, 0,
        0.00229071, -0.00454951, 0.00235324, 0.999984,
        -0.40903, 0.810125, -0.419955, 0.00550712};



    math::MathOperationsCpu mo;
    math::Matrix* A = host::NewReMatrixCopy(4, 4, Aa);
    math::Matrix* tq = host::NewReMatrixCopy(4, 4, tqa);
    math::Matrix* r = host::NewReMatrix(4, 4, 0);
    mo.multiply(r, tq, A);


    host::PrintReMatrix(r);
    floatt ra[] = {12331.1, 7512.76, 2523.71, 202.891,
        -4.54747e-13, 6185.88, 11962.2, 228.967,
        7.32392e-11, 6.72529e-11, 72.2064, 5397.05,
        -12.4107, -11.3961, -9.96319, -0.906733};

}

TEST_F(OglaMatrixTests, Det1) {



    floatt Aa[] = {
        75.2099, 0, 0, 2809.84, 0, 0, 2734.63, 0, 0, 2734.63, 0, 0, 2809.84, 0, 0, 75.2099,
        0, -5505.87, 0, 0, 36.605, 0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0,
        0, 0, -5505.87, 0, 0, 0, 0, 0, 36.605, 0, 0, 2734.63, 0, 0, 2734.63, 0,
        2734.63, 0, 0, -5505.87, 0, 0, 0, 0, 0, 0, 0, 0, 36.605, 0, 0, 2734.63,
        0, 36.605, 0, 0, -5505.87, 0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0,
        0, 0, 0, 0, 0, -10938.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2809.84, 0, 0, 75.2099, 0, 0, -5505.87, 0, 0, 36.605, 0, 0, 75.2099, 0, 0, 2809.84,
        0, 2734.63, 0, 0, 2734.63, 0, 0, -5505.87, 0, 0, 0, 0, 0, 36.605, 0, 0,
        0, 0, 36.605, 0, 0, 0, 0, 0, -5505.87, 0, 0, 2734.63, 0, 0, 2734.63, 0,
        2809.84, 0, 0, 75.2099, 0, 0, 36.605, 0, 0, -5505.87, 0, 0, 75.2099, 0, 0, 2809.84,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10938.5, 0, 0, 0, 0, 0,
        0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0, -5505.87, 0, 0, 36.605, 0,
        2734.63, 0, 0, 36.605, 0, 0, 0, 0, 0, 0, 0, 0, -5505.87, 0, 0, 2734.63,
        0, 2734.63, 0, 0, 2734.63, 0, 0, 36.605, 0, 0, 0, 0, 0, -5505.87, 0, 0,
        0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0, 36.605, 0, 0, -5505.87, 0,
        75.2099, 0, 0, 2809.84, 0, 0, 2734.63, 0, 0, 2734.63, 0, 0, 2809.84, 0, 0, 75.2099
    };

    math::Matrix* m = host::NewReMatrixCopy(16, 16, Aa);

    math::MathOperationsCpu mo;
    floatt d = 1;
    mo.det(&d, m);
    fprintf(stderr, "det == %f \n", d);


}

TEST_F(OglaMatrixTests, Det2) {



    floatt Aa[] = {
        -5505.87, 0.000000000000000, 0.000000000000000, 36.605000000000000, 0.000000000000000,
        0.000000000000000, -5505.870000000000000, 0.000000000000000, 0.000000000000000, 2734.630000000000000,
        0.000000000000000, 0.000000000000000, -5505.870000000000000, 0.000000000000000, 0.000000000000000,
        36.605000000000000, 0.000000000000000, 0.000000000000000, -5505.870000000000000, 0.000000000000000,
        0.000000000000000, 2809.840000000000000, 0.000000000000000, 0.000000000000000, 75.209900000000000
    };

    floatt a = -5505.87;

    fprintf(stderr, "dasd == %f \n", Aa[0]);
    fprintf(stderr, "dasd == %f \n", -5505.87);

    math::Matrix* m = host::NewReMatrixCopy(5, 5, Aa);

    host::PrintReMatrix("m ==", m);
    math::MathOperationsCpu mo;
    floatt d = 1;
    mo.det(&d, m);
    fprintf(stderr, "det == %f \n", d);


}

TEST_F(OglaMatrixTests, QRDet) {



    floatt Aa[] = {
        75.2099, 0, 0, 2809.84, 0, 0, 2734.63, 0, 0, 2734.63, 0, 0, 2809.84, 0, 0, 75.2099,
        0, -5505.87, 0, 0, 36.605, 0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0,
        0, 0, -5505.87, 0, 0, 0, 0, 0, 36.605, 0, 0, 2734.63, 0, 0, 2734.63, 0,
        2734.63, 0, 0, -5505.87, 0, 0, 0, 0, 0, 0, 0, 0, 36.605, 0, 0, 2734.63,
        0, 36.605, 0, 0, -5505.87, 0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0,
        0, 0, 0, 0, 0, -10938.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2809.84, 0, 0, 75.2099, 0, 0, -5505.87, 0, 0, 36.605, 0, 0, 75.2099, 0, 0, 2809.84,
        0, 2734.63, 0, 0, 2734.63, 0, 0, -5505.87, 0, 0, 0, 0, 0, 36.605, 0, 0,
        0, 0, 36.605, 0, 0, 0, 0, 0, -5505.87, 0, 0, 2734.63, 0, 0, 2734.63, 0,
        2809.84, 0, 0, 75.2099, 0, 0, 36.605, 0, 0, -5505.87, 0, 0, 75.2099, 0, 0, 2809.84,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10938.5, 0, 0, 0, 0, 0,
        0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0, -5505.87, 0, 0, 36.605, 0,
        2734.63, 0, 0, 36.605, 0, 0, 0, 0, 0, 0, 0, 0, -5505.87, 0, 0, 2734.63,
        0, 2734.63, 0, 0, 2734.63, 0, 0, 36.605, 0, 0, 0, 0, 0, -5505.87, 0, 0,
        0, 0, 2734.63, 0, 0, 0, 0, 0, 2734.63, 0, 0, 36.605, 0, 0, -5505.87, 0,
        75.2099, 0, 0, 2809.84, 0, 0, 2734.63, 0, 0, 2734.63, 0, 0, 2809.84, 0, 0, 75.2099
    };

    math::Matrix* A = host::NewReMatrixCopy(16, 16, Aa);
    math::Matrix* Q = host::NewReMatrix(16, 16);
    math::Matrix* R = host::NewReMatrix(16, 16);

    math::MathOperationsCpu mo;
    mo.qrDecomposition(Q, R, A);

    host::PrintReMatrix("A = ", A);
    host::PrintReMatrix("Q = ", Q);
    host::PrintReMatrix("R = ", R);

    fprintf(stderr, "det == %f \n", host::GetTrace(R));


}

TEST_F(OglaMatrixTests, Det3) {



    floatt Aa[] = {
        -5505.870000000000000, 0.000000000000000, 0.000000000000000, 36.605000000000000,
        0.000000000000000, -5505.870000000000000, 0.000000000000000, 0.000000000000000,
        0.000000000000000, 0.000000000000000, -5505.870000000000000, 0.000000000000000,
        36.605000000000000, 0.000000000000000, 0.000000000000000, -5505.870000000000000
    };

    math::Matrix* m = host::NewReMatrixCopy(4, 4, Aa);

    host::PrintReMatrix("m ==", m);
    math::MathOperationsCpu mo;
    floatt d = 1;
    mo.det(&d, m);
    fprintf(stderr, "det == %f \n", d);


}

TEST_F(OglaMatrixTests, QRDecomposition) {



    floatt Aa[] = {
        6, 5, 0,
        5, 1, 4,
        0, 4, 3
    };

    math::Matrix* A = host::NewReMatrixCopy(3, 3, Aa);
    math::Matrix* Q = host::NewReMatrix(3, 3);
    math::Matrix* R = host::NewReMatrix(3, 3);

    math::MathOperationsCpu mo;
    mo.qrDecomposition(Q, R, A);

    host::PrintReMatrix("A = ", A);
    host::PrintReMatrix("Q = ", Q);
    host::PrintReMatrix("R = ", R);

}

TEST_F(OglaMatrixTests, QRDecomposition1) {



    floatt Aa[] = {
        210165, 5397.29, -1.81985, 154.369,
        5399.38, 204582, 5755.02, 152.453,
        0, 5754.86, 209410, 214.621,
        0, 0, 70.4028, 204259
    };

    math::Matrix* A = host::NewReMatrixCopy(4, 4, Aa);
    math::Matrix* Q = host::NewReMatrix(4, 4);
    math::Matrix* R = host::NewReMatrix(4, 4);

    math::MathOperationsCpu mo;
    mo.qrDecomposition(Q, R, A);

    host::PrintReMatrix("A = ", A);
    host::PrintReMatrix("Q = ", Q);
    host::PrintReMatrix("R = ", R);

}