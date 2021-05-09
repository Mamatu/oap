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

#ifndef OAP_NETWORK_CUDA_API_H
#define OAP_NETWORK_CUDA_API_H

#include "oapNetworkGenericApi.h"

namespace oap
{

class NetworkCudaApi : public oap::NetworkGenericApi
{
  public:
    virtual ~NetworkCudaApi();

    virtual void setReValue (math::ComplexMatrix* matrix, uintt c, uintt r, floatt v) override;
    virtual void setHostWeights (oap::Layer& layer, math::ComplexMatrix* weights) override;
    virtual math::MatrixInfo getMatrixInfo (const math::ComplexMatrix* matrix) const override;
    virtual oap::Layer* createLayer (uintt neurons, bool hasBias, uintt samplesCount, Activation activation) override;
    virtual void copyKernelMatrixToKernelMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src) override;
    virtual void copyKernelMatrixToHostMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src) override;
    virtual void copyHostMatrixToKernelMatrix (math::ComplexMatrix* dst, const math::ComplexMatrix* src) override;
    virtual void deleteKernelMatrix (const math::ComplexMatrix* matrix) override;
    virtual math::ComplexMatrix* newKernelReMatrix (uintt columns, uintt rows) override;
    virtual math::ComplexMatrix* newKernelMatrixHostRef (const math::ComplexMatrix* matrix) override;
    virtual math::ComplexMatrix* newKernelMatrixKernelRef (const math::ComplexMatrix* matrix) override;
    virtual math::ComplexMatrix* newKernelSharedSubMatrix (const math::MatrixLoc& loc, const math::MatrixDim& mdim, const math::ComplexMatrix* matrix) override;
    virtual oap::Memory newKernelMemory (const oap::MemoryDim& dim) override;
    virtual math::ComplexMatrix* newKernelMatrixFromMatrixInfo (const math::MatrixInfo& minfo) override;
};
}
#endif
