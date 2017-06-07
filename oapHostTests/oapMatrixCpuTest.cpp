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
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "HostMatrixUtils.h"
#include "MathOperationsCpu.h"

class MatrixPtr : public std::unique_ptr<math::Matrix, void(*)(math::Matrix*)> {
private:
  static void DeleteMatrix(math::Matrix* matrix) {
    host::DeleteMatrix(matrix);
  }

public:
  MatrixPtr(math::Matrix* matrix) : std::unique_ptr<math::Matrix, void(*)(math::Matrix*)>(matrix, MatrixPtr::DeleteMatrix) {}
};

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
    MatrixPtr output = host::NewReMatrix(10, 10, 0);
    MatrixPtr matrix1 = host::NewReMatrix(10, 10, 1);
    MatrixPtr matrix2 = host::NewReMatrix(10, 10, 1);
    MatrixPtr eq_output = host::NewReMatrixCopy(10, 10, outputArray);
    mo.setSubColumns(5);
    mo.setSubRows(5);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output.get(), matrix1.get(), matrix2.get(), 5);

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
    MatrixPtr matrix1 = host::NewReMatrixCopy(4, 4, array);
    MatrixPtr matrix2 = host::NewReMatrixCopy(4, 4, array);
    MatrixPtr eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    MatrixPtr output = host::NewReMatrix(4, 4);
    math::AdditionOperationCpu additionOperation;
    additionOperation.setThreadsCount(m_threadsCount);
    additionOperation.setOutputMatrix(output.get());
    additionOperation.setMatrix1(matrix1.get());
    additionOperation.setMatrix2(matrix2.get());
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
    MatrixPtr matrix1 = host::NewReMatrixCopy(4, 4, array);
    MatrixPtr matrix2 = host::NewReMatrixCopy(4, 4, array);
    MatrixPtr eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    MatrixPtr output = host::NewReMatrix(4, 4);
    math::SubstracionOperationCpu substractionOperation;
    substractionOperation.setThreadsCount(m_threadsCount);
    substractionOperation.setOutputMatrix(output.get());
    substractionOperation.setMatrix1(matrix1.get());
    substractionOperation.setMatrix2(matrix2.get());
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
    MatrixPtr matrix1 = host::NewReMatrixCopy(4, 4, array);
    MatrixPtr matrix2 = host::NewReMatrixCopy(4, 4, array);
    MatrixPtr output = host::NewReMatrix(4, 4);
    MatrixPtr eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    math::MathOperationsCpu mo;
    mo.setThreadsCount(m_threadsCount);
    mo.setSubColumns(2);
    mo.add(output.get(), matrix1.get(), matrix2.get());

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

    MatrixPtr matrix1 = host::NewReMatrixCopy(5, 5, array);
    MatrixPtr matrix2 = host::NewReMatrixCopy(5, 5, array1);
    MatrixPtr output = host::NewReMatrix(5, 5);
    MatrixPtr eq_output = host::NewReMatrixCopy(5, 5, outputArray);
    math::DotProductOperationCpu multiplicationOperation;
    multiplicationOperation.setThreadsCount(m_threadsCount);
    multiplicationOperation.setOutputMatrix(output.get());
    multiplicationOperation.setSubColumns(2);
    multiplicationOperation.setMatrix1(matrix1.get());
    multiplicationOperation.setMatrix2(matrix2.get());
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

    MatrixPtr matrix1 = host::NewReMatrixCopy(3, 3, array2);
    MatrixPtr matrix2 = host::NewReMatrixCopy(3, 3, array3);
    MatrixPtr output = host::NewReMatrix(3, 3);
    MatrixPtr eq_output = host::NewReMatrix(3, 3);
    math::DiagonalizationOperationCpu diagonalizationOperation;
    diagonalizationOperation.setThreadsCount(m_threadsCount);
    diagonalizationOperation.setOutputMatrix(output.get());
    diagonalizationOperation.setMatrix1(matrix1.get());
    diagonalizationOperation.setMatrix2(matrix2.get());
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
    MatrixPtr matrix1 = host::NewReMatrixCopy(2, 2, array);
    MatrixPtr matrix2 = host::NewReMatrixCopy(2, 2, array1);
    MatrixPtr output = host::NewReMatrix(4, 4);
    MatrixPtr eq_output = host::NewReMatrixCopy(4, 4, outputArray);
    math::TensorProductOperationCpu tpOperation;
    tpOperation.setThreadsCount(m_threadsCount);
    tpOperation.setOutputMatrix(output.get());
    tpOperation.setMatrix1(matrix1.get());
    tpOperation.setMatrix2(matrix2.get());
    tpOperation.start();

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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
    MatrixPtr matrix1 = host::NewReMatrix(10, 10, 1);
    MatrixPtr matrix2 = host::NewReMatrix(10, 10, 2);
    MatrixPtr output = host::NewReMatrix(10, 10);
    MatrixPtr eq_output = host::NewReMatrixCopy(10, 10, outputArray);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output.get(), matrix1.get(), matrix2.get());

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
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

    MatrixPtr eq_output = host::NewReMatrixCopy(10, 10, outputArray);
    MatrixPtr matrix1 = host::NewReMatrix(10, 10, 1);
    MatrixPtr matrix2 = NULL;
    floatt m2 = 2.f;
    MatrixPtr output = host::NewReMatrix(10, 10);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output.get(), matrix1.get(), &m2);

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}

TEST_F(OapMatrixTests, SubMultiplication) {

    math::MathOperationsCpu mo;
    MatrixPtr output = host::NewReMatrix(10, 10, 0);
    MatrixPtr eq_output = host::NewReMatrix(10, 10, 0);
    eq_output->reValues[0] = 10;
    MatrixPtr matrix1 = host::NewReMatrix(10, 10, 1);
    MatrixPtr matrix2 = host::NewReMatrix(10, 10, 1);
    mo.setSubRows(1);
    mo.setSubColumns(1);
    mo.setThreadsCount(m_threadsCount);
    mo.multiply(output.get(), matrix1.get(), matrix2.get());

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
    MatrixPtr output = host::NewReMatrix(5, 10, 0);
    MatrixPtr eq_output = host::NewReMatrixCopy(5, 10, outputArray);
    MatrixPtr matrix1 = host::NewReMatrix(10, 5, 1);
    MatrixPtr matrix2 = NULL;
    mo.setSubRows(4);
    mo.setThreadsCount(m_threadsCount);
    mo.transpose(output.get(), matrix1.get());

    EXPECT_THAT(output.get(), MatrixIsEqual(eq_output.get()));
}
