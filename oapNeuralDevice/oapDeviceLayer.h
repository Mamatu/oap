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

#ifndef OAP_NEURAL_DEVICE_LAYER_H
#define OAP_NEURAL_DEVICE_LAYER_H

#include "oapLayer.h"

namespace oap
{

namespace math
{
  using Matrix = ::math::Matrix;
  using MatrixInfo = ::math::MatrixInfo;
}

class DeviceLayer : public Layer
{
  public:
    DeviceLayer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation);
    virtual ~DeviceLayer();

    math::MatrixInfo getOutputsInfo () const override;

    math::MatrixInfo getInputsInfo () const override;

    void getOutputs (math::Matrix* matrix, ArgType type) const override;

    void getHostWeights (math::Matrix* output) override;

    void setHostInputs(const math::Matrix* hInputs) override;

    void setDeviceInputs(const math::Matrix* dInputs) override;

    math::MatrixInfo getWeightsInfo () const override;

    void printHostWeights (const bool newLine) const override;

    void setHostWeights (math::Matrix* weights) override;

    void setDeviceWeights (math::Matrix* weights) override;
};
}
#endif
