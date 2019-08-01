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

#ifndef OAP_NEURAL_LAYER_H
#define OAP_NEURAL_LAYER_H

#define PRINT_CUMATRIX(m) logInfo ("%s %p %s %s", #m, m, oap::cuda::to_string(m).c_str(), oap::cuda::GetMatrixInfo(m).toString().c_str());
#define PRINT_MATRIX(m) logInfo ("%s %p %s %s", #m, m, oap::host::to_string(m).c_str(), oap::host::GetMatrixInfo(m).toString().c_str());

#include "ByteBuffer.h"

#include "CuProceduresApi.h"

#include "oapHostMatrixUPtr.h"
#include "oapDeviceMatrixUPtr.h"

class Network;

enum class Activation
{
  LINEAR,
  SIGMOID,
  TANH,
  SIN
};

enum class ArgType
{
  HOST,
  DEVICE,
  DEVICE_COPY,
};

enum class CalculationType
{
  HOST,
  DEVICE
};

class Layer
{
private:
  friend class Network;
  math::Matrix* m_inputs = nullptr;
  math::Matrix* m_tinputs = nullptr;
  math::Matrix* m_sums = nullptr;
  math::Matrix* m_errors = nullptr;
  math::Matrix* m_errorsAcc = nullptr;
  math::Matrix* m_errorsAux = nullptr;
  math::Matrix* m_errorsHost = nullptr;
  math::Matrix* m_weights = nullptr;
  math::Matrix* m_tweights = nullptr;
  math::Matrix* m_weights1 = nullptr;
  math::Matrix* m_weights2 = nullptr;
  math::Matrix* m_vec = nullptr;

  size_t m_neuronsCount = 0;
  const Layer* m_nextLayer = nullptr;

  size_t m_biasCount = 0;

  inline size_t getTotalNeuronsCount () const
  {
    return m_neuronsCount + m_biasCount;
  }

  std::pair<size_t, size_t> m_weightsDim;

  static void deallocate(math::Matrix** matrix);
  void checkHostInputs(const math::Matrix* hostInputs);

  friend class Network;

  Activation m_activation;

public:
  Layer(const Activation& activation = Activation::SIGMOID, bool addBias = false);

  ~Layer();

  inline Activation getActivation () const
  {
    return m_activation;
  }

  math::MatrixInfo getOutputsInfo () const;
  math::MatrixInfo getInputsInfo () const;
  void getOutputs (math::Matrix* matrix, oap::Type type) const;

  void setHostInputs(const math::Matrix* hInputs);
  void setDeviceInputs(const math::Matrix* dInputs);

  math::Matrix* getHostOutputs(math::Matrix* hInputs);
  math::Matrix* getDeviceOutputs(math::Matrix* dInputs);

  void allocateNeurons(size_t neuronsCount);

  void allocateWeights(const Layer* nextLayer);

  void deallocate();

  math::MatrixInfo getWeightsInfo () const;

  void setHostWeights (math::Matrix* weights);

  void getHostWeights (math::Matrix* output);

  void printHostWeights (bool newLine = false);

  size_t getNeuronsCount() const
  {
    return m_neuronsCount;
  }

  void setDeviceWeights (math::Matrix* weights);

  void initRandomWeights (const Layer* nextLayer);

  using RandCallback = std::function<floatt(size_t c, size_t r, floatt value)>;
  static std::unique_ptr<math::Matrix, std::function<void(const math::Matrix*)>> createRandomMatrix(size_t columns, size_t rows, RandCallback&& randCallback = [](size_t c, size_t r, floatt value){ return value; });

  void save (utils::ByteBuffer& buffer) const;
  static Layer* load (const utils::ByteBuffer& buffer);

  bool operator== (const Layer& layer) const;
  bool operator!= (const Layer& layer) const;
};

#endif
