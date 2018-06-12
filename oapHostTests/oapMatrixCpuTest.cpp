/*
 * Copyright 2016 - 2018 Marcin Matula
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
#include "MatchersUtils.h"
#include "oapHostMatrixUtils.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUPtr.h"


class OapMatrixTests : public testing::Test {
public:
    floatt eq_value;
    floatt value;
    int m_threadsCount;

    virtual void SetUp() {
        m_threadsCount = 1;
    }

    virtual void TearDown() {
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
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(10, 10, 0);
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrix(10, 10, 1);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrix(10, 10, 1);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(10, 10, outputArray);
    mo.setSubColumns(5);
    mo.setSubRows(5);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output, matrix1, matrix2, 5);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    math::AdditionOperationCpu additionOperation;
    additionOperation.setThreadsCount(m_threadsCount);
    additionOperation.setOutputMatrix(output);
    additionOperation.setMatrix1(matrix1);
    additionOperation.setMatrix2(matrix2);
    additionOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    math::SubstracionOperationCpu substractionOperation;
    substractionOperation.setThreadsCount(m_threadsCount);
    substractionOperation.setOutputMatrix(output);
    substractionOperation.setMatrix1(matrix1);
    substractionOperation.setMatrix2(matrix2);
    substractionOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
    math::MathOperationsCpu mo;
    mo.setThreadsCount(m_threadsCount);
    mo.setSubColumns(2);
    mo.add(output, matrix1, matrix2);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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

    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(5, 5, array);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(5, 5, array1);
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(5, 5);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(5, 5, outputArray);
    math::DotProductOperationCpu multiplicationOperation;
    multiplicationOperation.setThreadsCount(m_threadsCount);
    multiplicationOperation.setOutputMatrix(output);
    multiplicationOperation.setSubColumns(2);
    multiplicationOperation.setMatrix1(matrix1);
    multiplicationOperation.setMatrix2(matrix2);
    multiplicationOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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

    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(3, 3, array2);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(3, 3, array3);
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(3, 3);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrix(3, 3);
    math::DiagonalizationOperationCpu diagonalizationOperation;
    diagonalizationOperation.setThreadsCount(m_threadsCount);
    diagonalizationOperation.setOutputMatrix(output);
    diagonalizationOperation.setMatrix1(matrix1);
    diagonalizationOperation.setMatrix2(matrix2);
    diagonalizationOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(2, 2, array);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(2, 2, array1);
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
    math::TensorProductOperationCpu tpOperation;
    tpOperation.setThreadsCount(m_threadsCount);
    tpOperation.setOutputMatrix(output);
    tpOperation.setMatrix1(matrix1);
    tpOperation.setMatrix2(matrix2);
    tpOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}

TEST_F(OapMatrixTests, Multiplication1) {
    math::MathOperationsCpu mo;
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
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrix(10, 10, 1);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrix(10, 10, 2);
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(10, 10);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(10, 10, outputArray);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output, matrix1, matrix2);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}

TEST_F(OapMatrixTests, MultiplicationConst) {
    math::MathOperationsCpu mo;
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

    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(10, 10, outputArray);
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrix(10, 10, 1);
    oap::HostMatrixUPtr matrix2 = NULL;
    floatt m2 = 2.f;
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(10, 10);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output, matrix1, &m2);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}

TEST_F(OapMatrixTests, SubMultiplication) {

    math::MathOperationsCpu mo;
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(10, 10, 0);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrix(10, 10, 0);
    eq_output->reValues[0] = 10;
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrix(10, 10, 1);
    oap::HostMatrixUPtr matrix2 = oap::host::NewReMatrix(10, 10, 1);
    mo.setSubRows(1);
    mo.setSubColumns(1);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output, matrix1, matrix2);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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
    oap::HostMatrixUPtr output = oap::host::NewReMatrix(5, 10, 0);
    oap::HostMatrixUPtr eq_output = oap::host::NewReMatrixCopy(5, 10, outputArray);
    oap::HostMatrixUPtr matrix1 = oap::host::NewReMatrix(10, 5, 1);
    oap::HostMatrixUPtr matrix2 = NULL;
    mo.setSubRows(4);
    mo.setThreadsCount(m_threadsCount);
    mo.transpose(output, matrix1);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}
