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

#ifndef OAP_NEURAL_LAYER_GENERIC_H
#define OAP_NEURAL_LAYER_GENERIC_H

#include "ByteBuffer.h"

#include "Matrix.h"
#include "MatrixInfo.h"
#include "oapLayerStructure.h"

class Network;

namespace oap
{

class Layer
{
  template<typename CDst, typename CSrc, typename Get>
  static void cleanIterate (CDst& dst, const CSrc& src, Get&& get)
  {
    dst.clear();
    for (const auto& ep : src)
    {
      dst.push_back (get(ep));
    }
  }

  public:
    using Matrices = std::vector<math::Matrix*>;

    Layer (uintt neuronsCount, uintt biasesCount, uintt samplesCount, Activation activation);
    virtual ~Layer();

    uintt getTotalNeuronsCount() const;
    uintt getNeuronsCount() const;
    uintt getBiasesCount() const;
    uintt getSamplesCount() const;
    uintt getRowsCount() const;

    BPMatrices* getBPMatrices (uintt idx = 0) const;
    FPMatrices* getFPMatrices (uintt idx = 0) const;

    uintt getBPMatricesCount () const;
    uintt getFPMatricesCount () const;

    void addBPMatrices (BPMatrices* bpMatrices);
    void addFPMatrices (FPMatrices* fpMatrices);

    template<typename BPMatricesVec>
    void setBPMatrices (BPMatricesVec&& bpMatrices)
    {
      m_bpMatrices = std::forward<BPMatricesVec>(bpMatrices);
      cleanIterate(m_weights, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_weights;});
      cleanIterate(m_weights1, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_weights1;});
      cleanIterate(m_weights2, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_weights2;});
      cleanIterate(m_tinputs, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_tinputs;});
      cleanIterate(m_tweights, m_bpMatrices, [](const BPMatrices& bp){ return bp.m_tweights;});
    }

    void setBPMatrices (BPMatrices* bpMatrices);

    template<typename FPMatricesVec>
    void setFPMatrices (FPMatricesVec&& fpMatrices)
    {
      m_fpMatrices = std::forward<FPMatricesVec>(fpMatrices);
      cleanIterate(m_sums, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_sums;});
      cleanIterate(m_sums_wb, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_sums_wb;});
      cleanIterate(m_errors, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_errors;});
      cleanIterate(m_errors_wb, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_errors_wb;});
      cleanIterate(m_errorsAux, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_errorsAux;});
      cleanIterate(m_inputs, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_inputs;});
      cleanIterate(m_inputs_wb, m_fpMatrices, [](const FPMatrices& fp){ return fp.m_inputs_wb;});
    }

    void setFPMatrices (FPMatrices* fpMatrices);

    void setNextLayer (Layer* nextLayer);
    Layer* getNextLayer () const;

    Activation getActivation () const;

    virtual math::MatrixInfo getOutputsInfo () const = 0;
    virtual math::MatrixInfo getInputsInfo () const = 0;

    virtual void getOutputs (math::Matrix* matrix, ArgType type) const = 0;
    virtual void getHostWeights (math::Matrix* output) = 0;

    virtual void setHostInputs (const math::Matrix* hInputs) = 0;
    virtual void setDeviceInputs (const math::Matrix* dInputs) = 0;

    virtual math::MatrixInfo getWeightsInfo () const = 0;

    virtual void printHostWeights (bool newLine) const = 0;

    virtual void setHostWeights (math::Matrix* weights) = 0;
    virtual void setDeviceWeights (math::Matrix* weights) = 0;

    Matrices& getSums() { return m_sums; }
    Matrices& getSumsWB() { return m_sums_wb; }
    Matrices& getErrors() { return m_errors; }
    Matrices& getErrorsWB() { return m_errors_wb; }
    Matrices& getErrorsAux() { return m_errorsAux; }
    Matrices& getInputs() { return m_inputs; }
    Matrices& getInputsWB() { return m_inputs_wb; }
    Matrices& getTInputs() { return m_tinputs; }
    Matrices& getWeights() { return m_weights; }
    Matrices& getTWeights() { return m_tweights; }
    Matrices& getWeights1() { return m_weights1; }
    Matrices& getWeights2() { return m_weights2; }

    NBPair getNBPair() const
    {
      return std::make_pair(getNeuronsCount(), getBiasesCount());
    }
  private:
    Activation m_activation;
    uintt m_neuronsCount;
    uintt m_biasesCount;
    uintt m_samplesCount;


    Matrices m_sums;
    Matrices m_sums_wb;
    Matrices m_errors;
    Matrices m_errors_wb;
    Matrices m_errorsAux;
    Matrices m_inputs;
    Matrices m_inputs_wb;
    Matrices m_tinputs;
    Matrices m_weights;
    Matrices m_tweights;
    Matrices m_weights1;
    Matrices m_weights2;

    Layer* m_nextLayer = nullptr;
  protected:
    std::vector<FPMatrices*> m_fpMatrices;
    std::vector<BPMatrices*> m_bpMatrices;
};
}

#endif
