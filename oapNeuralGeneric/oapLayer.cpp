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

#ifndef OAP_LAYER_IMPL_H
#define OAP_LAYER_IMPL_H

#include "oapLayer.h"

#include <list>
#include "MatrixAPI.h"

namespace oap
{

Layer::Layer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation) :
  m_neuronsCount (neuronsCount), m_biasesCount (biasesCount), m_samplesCount(samplesCount), m_activation (activation)
{}

Layer::~Layer()
{}

uintt Layer::getTotalNeuronsCount() const
{
  return m_biasesCount + m_neuronsCount;
}

uintt Layer::getNeuronsCount() const
{
  return m_neuronsCount;
}

uintt Layer::getBiasesCount() const
{
  return m_biasesCount;
}

uintt Layer::getSamplesCount() const
{
  return m_samplesCount;
}

uintt Layer::getRowsCount() const
{
  return m_samplesCount * getTotalNeuronsCount ();
}

BPMatrices* Layer::getBPMatrices (uintt idx) const
{
  if (m_bpMatrices.empty())
  {
    return nullptr;
  }
  return m_bpMatrices[idx];
}

FPMatrices* Layer::getFPMatrices (uintt idx) const
{
  if (m_fpMatrices.empty())
  {
    return nullptr;
  }
  return m_fpMatrices[idx];
}

uintt Layer::getBPMatricesCount () const
{
  return m_bpMatrices.size();
}

uintt Layer::getFPMatricesCount () const
{
  return m_fpMatrices.size();
}

void Layer::addBPMatrices (BPMatrices* bpMatrices)
{
  if (bpMatrices == nullptr)
  {
    return;
  }
  m_bpMatrices.push_back (bpMatrices);
  m_weights.push_back (bpMatrices->m_weights);
  m_weights1.push_back (bpMatrices->m_weights1);
  m_weights2.push_back (bpMatrices->m_weights2);
  m_tinputs.push_back (bpMatrices->m_tinputs);
  m_tweights.push_back (bpMatrices->m_tweights);
}

void Layer::addFPMatrices (FPMatrices* fpMatrices)
{
  if (fpMatrices == nullptr)
  {
    return;
  }
  m_fpMatrices.push_back (fpMatrices);
  m_sums.push_back (fpMatrices->m_sums);
  m_sums_wb.push_back (fpMatrices->m_sums_wb);
  m_errors.push_back (fpMatrices->m_errors);
  m_errors_wb.push_back (fpMatrices->m_errors_wb);
  m_errorsAux.push_back (fpMatrices->m_errorsAux);
  m_inputs.push_back (fpMatrices->m_inputs);
  m_inputs_wb.push_back (fpMatrices->m_inputs_wb);
}

void Layer::setBPMatrices (BPMatrices* bpMatrices)
{
  addBPMatrices (bpMatrices);
}

void Layer::setFPMatrices (FPMatrices* fpMatrices)
{
  addFPMatrices (fpMatrices);
}

void Layer::setNextLayer (Layer* nextLayer)
{
  m_nextLayer = nextLayer;
}

Layer* Layer::getNextLayer () const
{
  return m_nextLayer;
}

Activation Layer::getActivation () const
{
  return m_activation;
}
}

#endif
