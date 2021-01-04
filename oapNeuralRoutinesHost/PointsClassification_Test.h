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

#ifndef OAP_POINTS_CLASSIFICATION__TEST_H
#define OAP_POINTS_CLASSIFICATION__TEST_H

#include "Math.h"
#include "oapProcedures.h"
#include "oapNetworkGenericApi.h"

namespace oap
{
  template<typename CopyHostMatrixToKernelMatrix, typename GetMatrixInfo>
  void runPointsClassification (uintt seed, oap::generic::SingleMatrixProcedures* singleApi, oap::generic::MultiMatricesProcedures* multiApi, oap::NetworkGenericApi* nga,
       CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix, GetMatrixInfo&& getMatrixInfo);
}

#include "PointsClassification_Test_impl.h"

#endif
