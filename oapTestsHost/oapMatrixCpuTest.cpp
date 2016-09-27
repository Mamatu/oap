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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "MockUtils.h"
#include "HostMatrixModules.h"
#include "MathOperationsCpu.h"

class OapMatrixTests : public testing::Test {
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
            EXPECT_THAT(output, MatrixIsEqual(eq_output));
        } else {
            EXPECT_EQ(eq_value, value);
        }
        host::DeleteMatrix(matrix1);
        host::DeleteMatrix(matrix2);
        host::DeleteMatrix(output);
        host::DeleteMatrix(eq_output);
    }
};

TEST_F(OapMatrixTests, SubMultiplication1) {
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

TEST_F(OapMatrixTests, Addition) {
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
    output = host::NewReMatrix(4, 4);
    math::AdditionOperationCpu additionOperation;
    additionOperation.setThreadsCount(tc);
    additionOperation.setOutputMatrix(output);
    additionOperation.setMatrix1(matrix1);
    additionOperation.setMatrix2(matrix2);
    additionOperation.start();
}

TEST_F(OapMatrixTests, Substraction) {
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
    output = host::NewReMatrix(4, 4);
    math::SubstracionOperationCpu substractionOperation;
    substractionOperation.setThreadsCount(tc);
    substractionOperation.setOutputMatrix(output);
    substractionOperation.setMatrix1(matrix1);
    substractionOperation.setMatrix2(matrix2);
    substractionOperation.start();
}

TEST_F(OapMatrixTests, Addition1) {

    floatt array[] = {1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0};
    floatt outputArray[] = {2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0};
    matrix1 = host::NewReMatrixCopy(4, 4, array);
    matrix2 = host::NewReMatrixCopy(4, 4, array);
    output = host::NewReMatrix(4, 4);
    eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    math::MathOperationsCpu mo;
    mo.setThreadsCount(tc);
    mo.setSubColumns(2);
    mo.add(output, matrix1, matrix2);
}

TEST_F(OapMatrixTests, Multiplication) {


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
    output = host::NewReMatrix(5, 5);
    eq_output = host::NewReMatrixCopy(5, 5, outputArray);
    math::DotProductOperationCpu multiplicationOperation;
    multiplicationOperation.setThreadsCount(tc);
    multiplicationOperation.setOutputMatrix(output);
    multiplicationOperation.setSubColumns(2);
    multiplicationOperation.setMatrix1(matrix1);
    multiplicationOperation.setMatrix2(matrix2);
    multiplicationOperation.start();
}

TEST_F(OapMatrixTests, Diagonalization) {
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
    output = host::NewReMatrix(3, 3);
    eq_output = host::NewReMatrix(3, 3);
    math::DiagonalizationOperationCpu diagonalizationOperation;
    diagonalizationOperation.setThreadsCount(tc);
    diagonalizationOperation.setOutputMatrix(output);
    diagonalizationOperation.setMatrix1(matrix1);
    diagonalizationOperation.setMatrix2(matrix2);
    diagonalizationOperation.start();
}

TEST_F(OapMatrixTests, TensorProduct) {

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
    output = host::NewReMatrix(4, 4);
    eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    math::TensorProductOperationCpu tpOperation;
    tpOperation.setThreadsCount(tc);
    tpOperation.setOutputMatrix(output);
    tpOperation.setMatrix1(matrix1);
    tpOperation.setMatrix2(matrix2);
    tpOperation.start();
}

TEST_F(OapMatrixTests, Multiplication1) {
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

TEST_F(OapMatrixTests, MultiplicationConst) {
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

TEST_F(OapMatrixTests, SubMultiplication) {

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

TEST_F(OapMatrixTests, Transpose) {
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
