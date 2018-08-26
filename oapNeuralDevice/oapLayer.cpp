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

Layer::Layer() : m_inputs(nullptr), m_sums(nullptr), m_errors(nullptr), m_weights(nullptr), m_weights1(nullptr), m_weights2(nullptr), m_neuronsCount(0)
{}

Layer::~Layer()
{
  deallocate();
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
  m_neuronsCount = neuronsCount;
  m_inputs = oap::cuda::NewDeviceReMatrix (1, m_neuronsCount);
  m_sums = oap::cuda::NewDeviceMatrixDeviceRef (m_inputs);
  m_errors = oap::cuda::NewDeviceMatrixDeviceRef (m_inputs);
}

void Layer::allocateWeights(const Layer* nextLayer)
{
  m_weights = oap::cuda::NewDeviceReMatrix (m_neuronsCount, nextLayer->m_neuronsCount);
  m_weights1 = oap::cuda::NewDeviceMatrixDeviceRef (m_weights);
  m_weights2 = oap::cuda::NewDeviceMatrixDeviceRef (m_weights);
  m_weightsDim = std::make_pair(m_neuronsCount, nextLayer->m_neuronsCount);

  initRandomWeights ();
}

void Layer::deallocate()
{
  deallocate (&m_inputs);
  deallocate (&m_sums);
  deallocate (&m_errors);
  deallocate (&m_weights);
  deallocate (&m_weights1);
  deallocate (&m_weights2);
}

void Layer::setHostWeights (math::Matrix* weights)
{
  oap::cuda::CopyHostMatrixToDeviceMatrix (m_weights, weights);
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
