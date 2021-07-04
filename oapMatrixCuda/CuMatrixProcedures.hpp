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
 * but WITHOUT ANY WARRANTY; without even the implied
 *warranthreadIndexY of
 * MERCHANTABILIthreadIndexY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OAP_CU_MATRIXPROCEDURES_H
#define OAP_CU_MATRIXPROCEDURES_H

#include "CuCore.hpp"

#include "CuMatrixUtils.hpp"
#include <stdio.h>
#include "Matrix.hpp"
#include "MatrixEx.hpp"

#include "CuProcedures/CuAddDotProductProcedures.hpp"
#include "CuProcedures/CuCompareProcedures.hpp"
#include "CuProcedures/CuCompareOptProcedures.hpp"
#include "CuProcedures/CuCompareOptProcedures2.hpp"
#include "CuProcedures/CuCopyProcedures.hpp"
#include "CuProcedures/CuCrossEntropyProcedures.hpp"
#include "CuProcedures/CuInversionProcedures.hpp"
#include "CuProcedures/CuDotProductSpecificProcedures.hpp"
#include "CuProcedures/CuDotProductSharedProcedures.hpp"
#include "CuProcedures/CuDotProductDimProcedures.hpp"
#include "CuProcedures/CuDotProductPeriodicProcedures.hpp"
#include "CuProcedures/CuDotProductDimPeriodicProcedures.hpp"
#include "CuProcedures/CuMultiplicationProcedures.hpp"
#include "CuProcedures/CuAdditionProcedures.hpp"
#include "CuProcedures/CuSubtractionProcedures.hpp"
#include "CuProcedures/CuAddSubtractionProcedures.hpp"
#include "CuProcedures/CuConjugateTransposeProcedures.hpp"
#include "CuProcedures/CuTransposeProcedures.hpp"
#include "CuProcedures/CuIdentityProcedures.hpp"
#include "CuProcedures/CuQRProcedures_GR.hpp"
#include "CuProcedures/CuQRProcedures_HT.hpp"
#include "CuProcedures/CuIsUpperTriangularProcedures.hpp"
#include "CuProcedures/CuMagnitudeOptProcedures.hpp"
#include "CuProcedures/CuMagnitudeOptProcedures2.hpp"
#include "CuProcedures/CuTriangularH.hpp"
#include "CuProcedures/CuTensorProductProcedures.hpp"
#include "CuProcedures/CuTensorProductDimProcedures.hpp"
#include "CuProcedures/CuSigmoidDimProcedures.hpp"
#include "CuProcedures/CuTanhDimProcedures.hpp"
#include "CuProcedures/CuSinDimProcedures.hpp"
#include "CuProcedures/CuSoftplusDimProcedures.hpp"
#include "CuProcedures/CuSumProcedures.hpp"
#include "CuProcedures/CuSetMatrixProcedures.hpp"
#include "CuProcedures/CuHadamardProductProcedures.hpp"
#include "CuProcedures/CuPartialHadamardProductProcedures.hpp"
#include "CuProcedures/CuVectorUtils.hpp"

#include "CuProcedures/CuPReluDimProcedures.hpp"
#include "CuProcedures/CuPReluProcedures.hpp"

#include "CuProcedures/CuReluDimProcedures.hpp"
#include "CuProcedures/CuReluProcedures.hpp"

#include "CuProcedures/CuPoolingProcedures.hpp"
#include "CuProcedures/CuConvolutionProcedures.hpp"

#include "CuProcedures/GenericApi/CuDotProductProcedures.hpp"
#include "CuProcedures/GenericApi/CuAdditionProcedures.hpp"
#include "CuProcedures/GenericApi/CuSubtractionProcedures.hpp"
#include "CuProcedures/GenericApi/CuAdditionConstProcedures.hpp"
#include "CuProcedures/GenericApi/CuMultiplyConstProcedures.hpp"
#include "CuProcedures/GenericApi/CuHadamardProductProcedures.hpp"
#include "CuProcedures/GenericApi/CuPartialHadamardProductProcedures.hpp"
#include "CuProcedures/GenericApi/CuTensorProductProcedures.hpp"
#include "CuProcedures/GenericApi/CuTransposeProcedures.hpp"
#include "CuProcedures/GenericApi/CuSinProcedures.hpp"
#include "CuProcedures/GenericApi/CuTanhProcedures.hpp"
#include "CuProcedures/GenericApi/CuSigmoidProcedures.hpp"
#include "CuProcedures/GenericApi/CuSoftplusProcedures.hpp"
#include "CuProcedures/GenericApi/CuReluProcedures.hpp"
#include "CuProcedures/GenericApi/CuPReluProcedures.hpp"
#include "CuProcedures/GenericApi/CuLinearProcedures.hpp"

#endif /* DEVICE_H */
