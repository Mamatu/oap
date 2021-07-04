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

#include "oapHostLayer.hpp"

namespace oap
{
HostLayer::HostLayer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation) : oap::Layer (neuronsCount, biasesCount, samplesCount, activation)
{}

HostLayer::~HostLayer()
{}

math::MatrixInfo HostLayer::getOutputsInfo () const
{
  return oap::generic::getOutputsInfo (*this, oap::chost::GetMatrixInfo);
}

math::MatrixInfo HostLayer::getInputsInfo () const
{
  return oap::host::getInputsInfo (*this);
}

void HostLayer::getOutputs (math::ComplexMatrix* matrix, ArgType type) const
{
  return oap::host::getOutputs (*this, matrix, type);
}

void HostLayer::getHostWeights (math::ComplexMatrix* output)
{
  oap::chost::CopyHostMatrixToHostMatrix (output, this->getBPMatrices()->m_weights);
}

void HostLayer::setHostInputs(const math::ComplexMatrix* hInputs)
{
  oap::host::setHostInputs (*this, hInputs);
}

void HostLayer::setDeviceInputs(const math::ComplexMatrix* dInputs)
{
  oap::host::setDeviceInputs (*this, dInputs);
}

math::MatrixInfo HostLayer::getWeightsInfo () const
{
  return oap::chost::GetMatrixInfo (this->getBPMatrices()->m_weights);
}

void HostLayer::printHostWeights (bool newLine) const
{
  oap::generic::printHostWeights (*this, newLine, oap::chost::CopyHostMatrixToHostMatrix);
}

void HostLayer::setHostWeights (math::ComplexMatrix* weights)
{
  oap::host::setHostWeights (*this, weights);
}

void HostLayer::setDeviceWeights (math::ComplexMatrix* weights)
{
  oap::host::setDeviceWeights (*this, weights);
}

}
