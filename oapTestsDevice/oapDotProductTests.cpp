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
#include "gtest/gtest.h"
#include "MatchersUtils.h"
#include "MatrixProcedures.h"
#include "MathOperationsCpu.h"
#include "HostMatrixModules.h"
#include "DeviceMatrixModules.h"
#include "KernelExecutor.h"

class OapMatrixCudaTests : public testing::Test {
public:
    math::Matrix* output;
    math::Matrix* eq_output;
    CuMatrix* cuMatrix;
    CUresult status;

    virtual void SetUp() {
        status = CUDA_SUCCESS;
        device::Context::Instance().create();
        output = NULL;
        eq_output = NULL;
        cuMatrix = new CuMatrix();
    }

    virtual void TearDown() {
        device::Context::Instance().destroy();
        delete cuMatrix;
        if (output != NULL && eq_output != NULL) {
            EXPECT_THAT(output, MatrixIsEqual(eq_output));
        }
        EXPECT_EQ(status, CUDA_SUCCESS);
        host::DeleteMatrix(output);
        host::DeleteMatrix(eq_output);
    }
};
