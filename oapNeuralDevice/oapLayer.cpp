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
  deallocate (&m_sums);
  deallocate (&m_errors);
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
  oap::HostMatrixUPtr matrix = oap::host::NewReMatrix (m_neuronsCount, m_nextLayerNeuronsCount);
  getHostWeights (matrix.get());
  std::stringstream sstream;
  sstream << "Layer (" << this << ") weights = ";
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
