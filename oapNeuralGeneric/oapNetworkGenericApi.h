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

#ifndef OAP_NETWORK_GENERIC_API_H
#define OAP_NETWORK_GENERIC_API_H

#include "Math.h"
#include "Matrix.h"

#include "oapLayer.h"

namespace oap
{
class NetworkGenericApi
{
  public:
    virtual ~NetworkGenericApi();

    virtual void setReValue (math::Matrix* matrix, uintt c, uintt r, floatt v) = 0;
    virtual void setHostWeights (Layer& layer, math::Matrix* weights) = 0;
    virtual math::MatrixInfo getMatrixInfo (const math::Matrix* matrix) const = 0;
    virtual FPMatrices* allocateFPMatrices (const Layer& layer, uintt samplesCount) = 0;
    virtual FPMatrices* allocateSharedFPMatrices (const Layer& layer, FPMatrices* orig) = 0;
    virtual BPMatrices* allocateBPMatrices (NBPair& pnb, NBPair& nnb) = 0;
    virtual void deallocateFPMatrices (FPMatrices* fpmatrices) = 0;
    virtual void deallocateBPMatrices (BPMatrices* bpmatrices) = 0;
    virtual Layer* createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation) = 0;
    virtual void copyKernelMatrixToKernelMatrix (math::Matrix* dst, const math::Matrix* src) = 0;
    virtual void copyKernelMatrixToHostMatrix (math::Matrix* dst, const math::Matrix* src) = 0;
    virtual void copyHostMatrixToKernelMatrix (math::Matrix* dst, const math::Matrix* src) = 0;
    virtual void deleteKernelMatrix (const math::Matrix* matrix) = 0;
    virtual math::Matrix* newKernelReMatrix (uintt columns, uintt rows) = 0;
    virtual math::Matrix* newKernelMatrixHostRef (math::Matrix* matrix) = 0;
    virtual math::Matrix* newKernelMatrixKernelRef (math::Matrix* matrix) = 0;
    virtual void connectLayers (oap::Layer* previous, oap::Layer* next) = 0;
};
}

#endif
