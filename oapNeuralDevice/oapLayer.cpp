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
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "oapLayer.h"

#include <list>
#include "MatrixAPI.h"

#include "oapGenericNetworkApi.h"

Layer::Layer () : m_lsPtr (new LayerS()), m_ls(*m_lsPtr)
{}

Layer::Layer(Activation activation, bool isbias) : m_lsPtr (new LayerS(activation, isbias)), m_ls(*m_lsPtr)
{}

Layer::~Layer()
{
  deallocate();
}

math::MatrixInfo Layer::getOutputsInfo () const
{
  return oap::generic::getOutputsInfo (m_ls);
}

math::MatrixInfo Layer::getInputsInfo () const
{
  return oap::generic::getOutputsInfo (m_ls);
}

void Layer::getOutputs (math::Matrix* matrix, ArgType type) const
{
  oap::generic::getOutputs (m_ls, matrix, type);
}

void Layer::setHostInputs(const math::Matrix* hInputs)
{
  oap::generic::setHostInputs (m_ls, hInputs);
}

void Layer::setDeviceInputs(const math::Matrix* dInputs)
{
  oap::generic::setDeviceInputs (m_ls, dInputs);
}

math::MatrixInfo Layer::getWeightsInfo () const
{
  return oap::generic::getWeightsInfo (m_ls);
}

void Layer::printHostWeights (bool newLine) const
{
  oap::generic::printHostWeights (m_ls, newLine);
}

void Layer::allocateNeurons(size_t neuronsCount)
{
  oap::generic::allocateNeurons (m_ls, neuronsCount, m_ls.m_biasCount);
}

void Layer::allocateWeights(const Layer* nextLayer)
{
  oap::generic::allocateWeights (m_ls, &nextLayer->m_ls);
  oap::generic::initRandomWeights (m_ls, &nextLayer->m_ls);
}

void Layer::deallocate()
{
  oap::generic::deallocate (m_ls);
  delete m_lsPtr;
}

void Layer::setHostWeights (math::Matrix* weights)
{
  oap::generic::setHostWeights (m_ls, weights);
}

void Layer::setDeviceWeights (math::Matrix* weights)
{
  oap::generic::setDeviceWeights (m_ls, weights);
}

void Layer::save (utils::ByteBuffer& buffer) const
{
  buffer.push_back (m_ls.getTotalNeuronsCount());

  buffer.push_back (m_ls.m_weightsDim.first);
  buffer.push_back (m_ls.m_weightsDim.second);

  oap::cuda::SaveMatrix (m_ls.m_inputs, buffer);;
  oap::cuda::SaveMatrix (m_ls.m_tinputs, buffer);
  oap::cuda::SaveMatrix (m_ls.m_sums, buffer);
  oap::cuda::SaveMatrix (m_ls.m_errors, buffer);
  oap::cuda::SaveMatrix (m_ls.m_errorsAcc, buffer);
  oap::cuda::SaveMatrix (m_ls.m_errorsAux, buffer);
  oap::cuda::SaveMatrix (m_ls.m_weights, buffer);
  oap::cuda::SaveMatrix (m_ls.m_tweights, buffer);
  oap::cuda::SaveMatrix (m_ls.m_weights1, buffer);
  oap::cuda::SaveMatrix (m_ls.m_weights2, buffer);
}

Layer* Layer::load (const utils::ByteBuffer& buffer)
{
  Layer* layer = new Layer ();

  layer->m_ls.m_neuronsCount = buffer.read <decltype (layer->m_ls.m_neuronsCount)>();

  layer->m_ls.m_weightsDim.first = buffer.read <decltype (layer->m_ls.m_weightsDim.first)>();
  layer->m_ls.m_weightsDim.second = buffer.read <decltype (layer->m_ls.m_weightsDim.second)>();

  layer->m_ls.m_inputs = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_tinputs = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_sums = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_errors = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_errorsAcc = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_errorsAux = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_weights = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_tweights = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_weights1 = oap::cuda::LoadMatrix (buffer);
  layer->m_ls.m_weights2 = oap::cuda::LoadMatrix (buffer);

  return layer;
}

bool Layer::operator== (const Layer& layer) const
{
  if (&layer == this)
  {
    return true;
  }

  if (m_ls.m_neuronsCount != layer.m_ls.m_neuronsCount)
  {
    return false;
  }

  if (m_ls.m_weightsDim.first != layer.m_ls.m_weightsDim.first)
  {
    return false;
  }

  if (m_ls.m_weightsDim.second != layer.m_ls.m_weightsDim.second)
  {
    return false;
  }

  oap::CuProceduresApi cuApi;

  std::list<std::pair<math::Matrix*, math::Matrix*>> list =
  {
    {m_ls.m_inputs, layer.m_ls.m_inputs},
    {m_ls.m_tinputs, layer.m_ls.m_tinputs},
    {m_ls.m_sums, layer.m_ls.m_sums},
    {m_ls.m_errors , layer.m_ls.m_errors },
    {m_ls.m_errorsAcc , layer.m_ls.m_errorsAcc },
    {m_ls.m_errorsAux , layer.m_ls.m_errorsAux },
    {m_ls.m_weights, layer.m_ls.m_weights},
    {m_ls.m_tweights, layer.m_ls.m_tweights},
    {m_ls.m_weights1, layer.m_ls.m_weights1},
    {m_ls.m_weights2, layer.m_ls.m_weights2}
  };

  for (auto& pair : list)
  {
    debugAssert ((pair.first != nullptr && pair.second != nullptr) || (pair.first == nullptr && pair.second == nullptr));
    if (pair.first != nullptr && pair.second != nullptr && !cuApi.compare (pair.first, pair.second))
    {
      oap::cuda::PrintMatrix ("pair.first = ", pair.first);
      oap::cuda::PrintMatrix ("pair.second = ", pair.second);
      return false;
    }
  }
  return true;
}

bool Layer::operator!= (const Layer& layer) const
{
  return !(*this == layer);
}
