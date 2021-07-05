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

#include "oapDeviceLayer.hpp"
#include "oapCudaMatrixUtils.hpp"
#include "oapDeviceNeuralApi.hpp"

namespace oap
{

DeviceLayer::DeviceLayer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation) : Layer (neuronsCount, biasesCount, samplesCount, activation)
{}

DeviceLayer::~DeviceLayer()
{}

math::MatrixInfo DeviceLayer::getOutputsInfo () const
{
  return oap::generic::getOutputsInfo (*this, oap::cuda::GetMatrixInfo);
}

math::MatrixInfo DeviceLayer::getInputsInfo () const
{
  return oap::device::getInputsInfo (*this);
}

void DeviceLayer::getOutputs (math::ComplexMatrix* matrix, ArgType type) const
{
  return oap::device::getOutputs (*this, matrix, type);
}

void DeviceLayer::getHostWeights (math::ComplexMatrix* output)
{
  oap::cuda::CopyDeviceMatrixToHostMatrix (output, getBPMatrices()->m_weights);
}

void DeviceLayer::setHostInputs(const math::ComplexMatrix* hInputs)
{
  oap::device::setHostInputs (*this, hInputs);
}

void DeviceLayer::setDeviceInputs(const math::ComplexMatrix* dInputs)
{
  oap::device::setDeviceInputs (*this, dInputs);
}

math::MatrixInfo DeviceLayer::getWeightsInfo () const
{
  return oap::cuda::GetMatrixInfo (getBPMatrices()->m_weights);
}

void DeviceLayer::printHostWeights (bool newLine) const
{
  oap::generic::printHostWeights (*this, newLine, oap::cuda::CopyDeviceMatrixToHostMatrix);
}

void DeviceLayer::setHostWeights (math::ComplexMatrix* weights)
{
  oap::device::setHostWeights (*this, weights);
}

void DeviceLayer::setDeviceWeights (math::ComplexMatrix* weights)
{
  oap::device::setDeviceWeights (*this, weights);
}

}
