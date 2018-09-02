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

#ifndef OAP_NEURAL_LAYER_H
#define OAP_NEURAL_LAYER_H

#include "CuProceduresApi.h"

#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixUPtr.h"

class Layer
{
public:
  math::Matrix* m_inputs;
  math::Matrix* m_tinputs;
  math::Matrix* m_sums;
  math::Matrix* m_tsums;
  math::Matrix* m_errors;
  math::Matrix* m_terrors;
  math::Matrix* m_weights;
  math::Matrix* m_tweights;
  math::Matrix* m_weights1;
  math::Matrix* m_weights2;

  size_t m_neuronsCount;

  std::pair<size_t, size_t> m_weightsDim;

  static void deallocate(math::Matrix** matrix);
public:
  Layer();

  ~Layer();

  void allocateNeurons(size_t neuronsCount);

  void allocateWeights(const Layer* nextLayer);

  void deallocate();

  void setHostWeights (math::Matrix* weights);

  void setDeviceWeights (math::Matrix* weights);

  void initRandomWeights();

  static std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> createRandomMatrix(size_t columns, size_t rows);
};

#endif
