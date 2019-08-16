/*
 * Copyright 2016 - 2019 Marcin Matula
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
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "CuProceduresApi.h"
#include "MathOperationsCpu.h"
#include "oapHostMatrixUtils.h"
#include "oapCudaMatrixUtils.h"
#include "KernelExecutor.h"

class OapCompareCudaTests : public testing::Test {
public:

    math::Matrix* output;
    math::Matrix* eq_output;
    oap::CuProceduresApi* cuMatrix;
    CUresult status;

    virtual void SetUp() {
        status = CUDA_SUCCESS;
        oap::cuda::Context::Instance().create();
        output = NULL;
        eq_output = NULL;
        cuMatrix = new oap::CuProceduresApi();
    }

    virtual void TearDown() {
        delete cuMatrix;
        if (output != NULL && eq_output != NULL) {
            EXPECT_THAT(output, MatrixIsEqual(eq_output));
        }
        oap::host::DeleteMatrix(output);
        oap::host::DeleteMatrix(eq_output);
        oap::cuda::Context::Instance().destroy();
    }
};

TEST_F(OapCompareCudaTests, CompareReMatrixTest1) {
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

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, 10, 10);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, 10, 10);
    oap::cuda::CopyHostArrayToDeviceMatrix(matrix1, hArray, NULL, 100);
    oap::cuda::CopyHostArrayToDeviceMatrix(matrix2, hArray, NULL, 100);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    floatt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    floatt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_TRUE(resultVer1);
    EXPECT_TRUE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTest1Fail) {
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

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, 10, 10);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, 10, 10);
    oap::cuda::CopyHostArrayToDeviceMatrix(matrix1, hArray1, NULL, 100);
    oap::cuda::CopyHostArrayToDeviceMatrix(matrix2, hArray2, NULL, 100);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    floatt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    floatt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_FALSE(resultVer1);
    EXPECT_FALSE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTestBigData) {

    uintt columns = 50;
    uintt rows = 32;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_TRUE(resultVer1);
    EXPECT_TRUE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTestBigData1) {

    uintt columns = 50;
    uintt rows = 50;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_TRUE(resultVer1);
    EXPECT_TRUE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTestBigData2) {

    uintt columns = 70;
    uintt rows = 70;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_TRUE(resultVer1);
    EXPECT_TRUE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTestBigData3) {

    uintt columns = 111;
    uintt rows = 111;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_TRUE(resultVer1);
    EXPECT_TRUE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTestBigData3Fail) {

    uintt columns = 111;
    uintt rows = 111;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    CudaUtils::SetReValue(matrix1, columns * rows - 1, 10);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_FALSE(resultVer1);
    EXPECT_FALSE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTestBigData4) {

    uintt columns = 1000;
    uintt rows = 1000;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_TRUE(resultVer1);
    EXPECT_TRUE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTestBigData4Fail) {

    uintt columns = 1000;
    uintt rows = 1000;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    CudaUtils::SetReValue(matrix1, columns * rows - 1, 10);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_FALSE(resultVer1);
    EXPECT_FALSE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}

TEST_F(OapCompareCudaTests, CompareReMatrixTest5Fail) {

    uintt columns = 11;
    uintt rows = 11;

    math::Matrix* matrix1 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);
    math::Matrix* matrix2 = oap::cuda::NewDeviceMatrix(true, false, columns, rows);

    CudaUtils::SetReValue(matrix1, columns * rows - 1, 10);

    bool resultVer1 = cuMatrix->compare(matrix1, matrix2);
    uintt outcomeVer1 = cuMatrix->getCompareOperationSum();
    bool resultVer2 = cuMatrix->compareVer2(matrix1, matrix2);
    uintt outcomeVer2 = cuMatrix->getCompareOperationSum();

    oap::cuda::DeleteDeviceMatrix(matrix1);
    oap::cuda::DeleteDeviceMatrix(matrix2);

    EXPECT_FALSE(resultVer1);
    EXPECT_FALSE(resultVer2);
    EXPECT_EQ(outcomeVer2, outcomeVer1);
}
