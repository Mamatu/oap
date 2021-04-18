/*
 * Copyright 2016 - 2021 Marcin Matula
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
#include "oapHostComplexMatrixUPtr.h"


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
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrixWithValue (10, 10, 0);
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (10, 10, 1);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixWithValue (10, 10, 1);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(10, 10, outputArray);
    mo.setSubColumns(5);
    mo.setSubRows(5);
    mo.setThreadsCount(m_threadsCount);

    oap::host::SetSubColumns (eq_output, 5);
    oap::host::SetSubRows (eq_output, 5);

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
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    math::AdditionOperationCpu additionOperation;
    additionOperation.setThreadsCount(m_threadsCount);
    additionOperation.setOutputMatrix(output);
    additionOperation.setMatrix1(matrix1);
    additionOperation.setMatrix2(matrix2);
    additionOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}

TEST_F(OapMatrixTests, Subtraction) {
    floatt array[] = {1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    floatt outputArray[] = {0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0};
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    math::SubstracionOperationCpu subtractionOperation;
    subtractionOperation.setThreadsCount(m_threadsCount);
    subtractionOperation.setOutputMatrix(output);
    subtractionOperation.setMatrix1(matrix1);
    subtractionOperation.setMatrix2(matrix2);
    subtractionOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}

TEST_F(OapMatrixTests, Addition1) {

    floatt array[] =
    {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0
    };
    floatt outputArray[] =
    {
      2, 0, 0, 0,
      0, 2, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0
    };
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(4, 4, array);
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
    math::MathOperationsCpu mo;
    mo.setThreadsCount(m_threadsCount);
    mo.setSubColumns(2);
    mo.add(output, matrix1, matrix2);

    oap::host::SetSubColumns (eq_output, 2);
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

    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(5, 5, array);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(5, 5, array1);
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(5, 5);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(5, 5, outputArray);
    math::DotProductOperationCpu multiplicationOperation;
    multiplicationOperation.setThreadsCount(m_threadsCount);
    multiplicationOperation.setOutputMatrix(output);

    multiplicationOperation.setSubColumns(2);

    multiplicationOperation.setMatrix1(matrix1);
    multiplicationOperation.setMatrix2(matrix2);
    multiplicationOperation.start();

    oap::host::SetSubColumns (eq_output, 2);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}
#if 0
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

    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(3, 3, array2);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(3, 3, array3);
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(3, 3);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrix(3, 3);
    math::DiagonalizationOperationCpu diagonalizationOperation;
    diagonalizationOperation.setThreadsCount(m_threadsCount);
    diagonalizationOperation.setOutputMatrix(output);
    diagonalizationOperation.setMatrix1(matrix1);
    diagonalizationOperation.setMatrix2(matrix2);
    diagonalizationOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}
#endif

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
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixCopy(2, 2, array);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixCopy(2, 2, array1);
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(4, 4);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(4, 4, outputArray);
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
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (10, 10, 1);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixWithValue (10, 10, 2);
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(10, 10);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(10, 10, outputArray);
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

    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(10, 10, outputArray);
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (10, 10, 1);
    oap::HostComplexMatrixUPtr matrix2 = nullptr;
    floatt m2 = 2.f;
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrix(10, 10);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output, matrix1, &m2);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}

TEST_F(OapMatrixTests, SubMultiplication) {

    math::MathOperationsCpu mo;
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrixWithValue (10, 10, 0);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixWithValue (10, 10, 0);
    *GetRePtrIndex (eq_output, 0) = 10;
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (10, 10, 1);
    oap::HostComplexMatrixUPtr matrix2 = oap::host::NewReMatrixWithValue (10, 10, 1);
    mo.setSubRows(1);
    mo.setSubColumns(1);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output, matrix1, matrix2);

    oap::host::SetSubs (eq_output, 1, 1);

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
    oap::HostComplexMatrixUPtr output = oap::host::NewReMatrixWithValue (5, 10, 0);
    oap::HostComplexMatrixUPtr eq_output = oap::host::NewReMatrixCopy(5, 10, outputArray);
    oap::HostComplexMatrixUPtr matrix1 = oap::host::NewReMatrixWithValue (10, 5, 1);
    oap::HostComplexMatrixUPtr matrix2 = nullptr;
    mo.setSubRows(4);
    mo.setThreadsCount(m_threadsCount);
    mo.transpose(output, matrix1);

    oap::host::SetSubRows (eq_output, 4);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}
