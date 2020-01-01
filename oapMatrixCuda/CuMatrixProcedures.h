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

#include "CuCore.h"

#include "CuMatrixUtils.h"
#include <stdio.h>
#include "Matrix.h"
#include "MatrixEx.h"

#include "CuProcedures/CuAddDotProductProcedures.h"
#include "CuProcedures/CuCompareProcedures.h"
#include "CuProcedures/CuCompareOptProcedures.h"
#include "CuProcedures/CuCompareOptProcedures2.h"
#include "CuProcedures/CuCopyProcedures.h"
#include "CuProcedures/CuCrossEntropyProcedures.h"
#include "CuProcedures/CuInversionProcedures.h"
#include "CuProcedures/CuDotProductSpecificProcedures.h"
#include "CuProcedures/CuDotProductSharedProcedures.h"
#include "CuProcedures/CuDotProductDimProcedures.h"
#include "CuProcedures/CuDotProductPeriodicProcedures.h"
#include "CuProcedures/CuDotProductDimPeriodicProcedures.h"
#include "CuProcedures/CuMultiplicationProcedures.h"
#include "CuProcedures/CuAdditionProcedures.h"
#include "CuProcedures/CuSubstractionProcedures.h"
#include "CuProcedures/CuAddSubstractionProcedures.h"
#include "CuProcedures/CuConjugateTransposeProcedures.h"
#include "CuProcedures/CuTransposeProcedures.h"
#include "CuProcedures/CuIdentityProcedures.h"
#include "CuProcedures/CuQRProcedures_GR.h"
#include "CuProcedures/CuQRProcedures_HT.h"
#include "CuProcedures/CuIsUpperTriangularProcedures.h"
#include "CuProcedures/CuMagnitudeOptProcedures.h"
#include "CuProcedures/CuMagnitudeOptProcedures2.h"
#include "CuProcedures/CuTriangularH.h"
#include "CuProcedures/CuTensorProductProcedures.h"
#include "CuProcedures/CuTensorProductDimProcedures.h"
#include "CuProcedures/CuSigmoidDimProcedures.h"
#include "CuProcedures/CuTanhDimProcedures.h"
#include "CuProcedures/CuSinDimProcedures.h"
#include "CuProcedures/CuSoftplusDimProcedures.h"
#include "CuProcedures/CuSumProcedures.h"
#include "CuProcedures/CuSetMatrixProcedures.h"
#include "CuProcedures/CuHadamardProductProcedures.h"
#include "CuProcedures/CuPartialHadamardProductProcedures.h"
#include "CuProcedures/CuVectorUtils.h"

#include "CuProcedures/CuPReluDimProcedures.h"
#include "CuProcedures/CuPReluProcedures.h"

#include "CuProcedures/CuReluDimProcedures.h"
#include "CuProcedures/CuReluProcedures.h"

#include "CuProcedures/CuPoolingProcedures.h"
#include "CuProcedures/CuConvolutionProcedures.h"

#endif /* DEVICE_H */
