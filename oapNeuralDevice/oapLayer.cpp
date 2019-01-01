/*
 * Copyright 2016 - 2018 Marcin Matula
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

void Layer::checkHostInputs(const math::Matrix* hostInputs)
{
  if (hostInputs->columns != 1)
  {
    throw std::runtime_error ("Columns of hostInputs matrix must be equal 1");
  }

  if (hostInputs->rows != m_neuronsCount)
  {
    throw std::runtime_error ("Rows of hostInputs matrix must be equal neurons count (or neurons count + 1 if is bias neuron)");
  }
}

Layer::Layer(bool hasBias) :
m_inputs(nullptr), m_tinputs(nullptr), m_sums(nullptr),
m_tsums(nullptr), m_errors(nullptr), m_terrors(nullptr),
m_weights(nullptr), m_tweights(nullptr), m_weights1(nullptr),
m_weights2(nullptr), m_neuronsCount(0), m_nextLayerNeuronsCount(0),
m_hasBias(hasBias)
{}

Layer::~Layer()
{
  deallocate();
}

void Layer::setHostInputs(const math::Matrix* hostInputs)
{
  checkHostInputs (hostInputs);

  if (m_hasBias)
  {
    hostInputs->reValues[m_neuronsCount - 1] = 1;
  }

  oap::cuda::CopyHostMatrixToDeviceMatrix (m_inputs, hostInputs);
}

void Layer::deallocate(math::Matrix** matrix)
{
  if (matrix != nullptr)
  {
    oap::cuda::DeleteDeviceMatrix (*matrix);
    matrix = nullptr;
  }
}

void Layer::allocateNeurons(size_t neuronsCount)
{
  debugInfo ("Layer %p allocates %lu neurons", this, neuronsCount);
  m_neuronsCount = m_hasBias ? neuronsCount + 1 : neuronsCount;

  m_inputs = oap::cuda::NewDeviceReMatrix (1, m_neuronsCount);
  m_sums = oap::cuda::NewDeviceMatrixDeviceRef (m_inputs);
  m_tsums = oap::cuda::NewDeviceMatrix (m_neuronsCount, 1);
  m_errors = oap::cuda::NewDeviceMatrixDeviceRef (m_inputs);
  m_terrors = oap::cuda::NewDeviceReMatrix (m_neuronsCount, 1); //todo: use transpose
  m_tinputs = oap::cuda::NewDeviceReMatrix (m_neuronsCount, 1); //todo: use transpose
}

void Layer::allocateWeights(const Layer* nextLayer)
{
  m_weights = oap::cuda::NewDeviceReMatrix (m_neuronsCount, nextLayer->m_neuronsCount);
  m_tweights = oap::cuda::NewDeviceReMatrix (nextLayer->m_neuronsCount, m_neuronsCount);
  m_weights1 = oap::cuda::NewDeviceMatrixDeviceRef (m_weights);
  m_weights2 = oap::cuda::NewDeviceMatrixDeviceRef (m_weights);
  m_weightsDim = std::make_pair(m_neuronsCount, nextLayer->m_neuronsCount);

  m_nextLayerNeuronsCount = nextLayer->m_neuronsCount;

  initRandomWeights ();
}

void Layer::deallocate()
{
  deallocate (&m_inputs);
  deallocate (&m_tinputs);
  deallocate (&m_sums);
  deallocate (&m_tsums);
  deallocate (&m_errors);
  deallocate (&m_terrors);
  deallocate (&m_weights);
  deallocate (&m_tweights);
  deallocate (&m_weights1);
  deallocate (&m_weights2);
}

void Layer::setHostWeights (math::Matrix* weights)
{
  oap::cuda::CopyHostMatrixToDeviceMatrix (m_weights, weights);
}

void Layer::getHostWeights (math::Matrix* output)
{
  oap::cuda::CopyDeviceMatrixToHostMatrix (output, m_weights);
}

void Layer::printHostWeights ()
{
  std::stringstream sstream;
  sstream << "Layer (" << this << ") weights = ";

  if (m_weights == nullptr)
  {
    oap::host::PrintMatrix (sstream.str(), nullptr);
    return;
  }

  oap::HostMatrixUPtr matrix = oap::host::NewReMatrix (m_neuronsCount, m_nextLayerNeuronsCount);
  getHostWeights (matrix.get());
  oap::host::PrintMatrix (sstream.str(), matrix.get());
}

void Layer::setDeviceWeights (math::Matrix* weights)
{
  oap::cuda::CopyDeviceMatrixToDeviceMatrix (m_weights, weights);
}

std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> Layer::createRandomMatrix(size_t columns, size_t rows)
{
  std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> randomMatrix(oap::host::NewReMatrix(columns, rows),
                  [](const math::Matrix* m){oap::host::DeleteMatrix(m);});

  std::random_device rd;
  std::default_random_engine dre (rd());
  std::uniform_real_distribution<> dis(0., 1.);

  for (size_t idx = 0; idx < columns * rows; ++idx)
  {
    randomMatrix->reValues[idx] = dis(dre);
  }

  return std::move (randomMatrix);
}

void Layer::initRandomWeights()
{
  if (m_weights == nullptr)
  {
    throw std::runtime_error("m_weights == nullptr");
  }

  auto randomMatrix = createRandomMatrix (m_weightsDim.first, m_weightsDim.second);
  oap::cuda::CopyHostMatrixToDeviceMatrix (m_weights, randomMatrix.get());
}

void Layer::save (utils::ByteBuffer& buffer) const
{
  buffer.push_back (m_neuronsCount);
  buffer.push_back (m_nextLayerNeuronsCount);

  buffer.push_back (m_weightsDim.first);
  buffer.push_back (m_weightsDim.second);

  buffer.push_back (m_hasBias);

  oap::cuda::SaveMatrix (m_inputs, buffer);;
  oap::cuda::SaveMatrix (m_tinputs, buffer);
  oap::cuda::SaveMatrix (m_sums, buffer);
  oap::cuda::SaveMatrix (m_tsums, buffer);
  oap::cuda::SaveMatrix (m_errors, buffer);
  oap::cuda::SaveMatrix (m_terrors, buffer);
  oap::cuda::SaveMatrix (m_weights, buffer);
  oap::cuda::SaveMatrix (m_tweights, buffer);
  oap::cuda::SaveMatrix (m_weights1, buffer);
  oap::cuda::SaveMatrix (m_weights2, buffer);
}

Layer* Layer::load (const utils::ByteBuffer& buffer)
{
  Layer* layer = new Layer ();

  layer->m_neuronsCount = buffer.read <decltype (layer->m_neuronsCount)>();
  layer->m_nextLayerNeuronsCount = buffer.read <decltype (layer->m_nextLayerNeuronsCount)>();

  layer->m_weightsDim.first = buffer.read <decltype (layer->m_weightsDim.first)>();
  layer->m_weightsDim.second = buffer.read <decltype (layer->m_weightsDim.second)>();

  layer->m_hasBias = buffer.read <decltype (layer->m_hasBias)>();

  layer->m_inputs = oap::cuda::LoadMatrix (buffer);
  layer->m_tinputs = oap::cuda::LoadMatrix (buffer);
  layer->m_sums = oap::cuda::LoadMatrix (buffer);
  layer->m_tsums = oap::cuda::LoadMatrix (buffer);
  layer->m_errors = oap::cuda::LoadMatrix (buffer);
  layer->m_terrors = oap::cuda::LoadMatrix (buffer);
  layer->m_weights = oap::cuda::LoadMatrix (buffer);
  layer->m_tweights = oap::cuda::LoadMatrix (buffer);
  layer->m_weights1 = oap::cuda::LoadMatrix (buffer);
  layer->m_weights2 = oap::cuda::LoadMatrix (buffer);

  return layer;
}

bool Layer::operator== (const Layer& layer) const
{
  if (&layer == this)
  {
    return true;
  }

  if (m_neuronsCount != layer.m_neuronsCount)
  {
    return false;
  }

  if (m_nextLayerNeuronsCount != layer.m_nextLayerNeuronsCount)
  {
    return false;
  }

  if (m_weightsDim.first != layer.m_weightsDim.first)
  {
    return false;
  }

  if (m_weightsDim.second != layer.m_weightsDim.second)
  {
    return false;
  }

  if (m_hasBias != layer.m_hasBias)
  {
    return false;
  }

  oap::CuProceduresApi cuApi;

  std::list<std::pair<math::Matrix*, math::Matrix*>> list =
  {
    {m_inputs, layer.m_inputs},
     {m_tinputs, layer.m_tinputs},
     {m_sums, layer.m_sums},
     {m_tsums, layer.m_tsums},
     {m_errors , layer.m_errors },
     {m_terrors, layer.m_terrors},
     {m_weights, layer.m_weights},
     {m_tweights, layer.m_tweights},
     {m_weights1, layer.m_weights1},
     {m_weights2, layer.m_weights2}
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
