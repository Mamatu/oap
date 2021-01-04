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

#ifndef OAP_NETWORK_HOST_API_H
#define OAP_NETWORK_HOST_API_H

#include "oapNetworkGenericApi.h"

namespace oap
{
class NetworkHostApi : public oap::NetworkGenericApi
{
  public:
    virtual ~NetworkHostApi();

    virtual void setReValue (math::Matrix* matrix, uintt c, uintt r, floatt v) override;
    virtual void setHostWeights (oap::Layer& layer, math::Matrix* weights) override;
    virtual math::MatrixInfo getMatrixInfo (const math::Matrix* matrix) const override;
    virtual FPMatrices* allocateFPMatrices (const Layer& layer, uintt samplesCount) override;
    virtual FPMatrices* allocateSharedFPMatrices (const Layer& layer, FPMatrices* orig) override;
    virtual BPMatrices* allocateBPMatrices (NBPair& pnb, NBPair& nnb) override;
    virtual void deallocateFPMatrices (FPMatrices* fpmatrices) override;
    virtual void deallocateBPMatrices (BPMatrices* bpmatrices) override;
    virtual oap::Layer* createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation) override;
    virtual void copyKernelMatrixToKernelMatrix (math::Matrix* dst, const math::Matrix* src) override;
    virtual void copyKernelMatrixToHostMatrix (math::Matrix* dst, const math::Matrix* src) override;
    virtual void copyHostMatrixToKernelMatrix (math::Matrix* dst, const math::Matrix* src) override;
    virtual void deleteKernelMatrix (const math::Matrix* matrix) override;
    virtual math::Matrix* newKernelReMatrix (uintt columns, uintt rows) override;
    virtual math::Matrix* newKernelMatrixHostRef (math::Matrix* matrix) override;
    virtual math::Matrix* newKernelMatrixKernelRef (math::Matrix* matrix) override;
    virtual void connectLayers (oap::Layer* previous, oap::Layer* next) override;
};
}
#endif
