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

#ifndef OAP_NEURAL_NETWORK_H
#define OAP_NEURAL_NETWORK_H

#include "oapLayer.hpp"
#include "oapProcedures.hpp"
#include "oapNetworkGenericApi.hpp"
#include "oapHostComplexMatrixPtr.hpp"
#include "oapHostComplexMatrixUPtr.hpp"
#include "oapNetworkTopology.hpp"

namespace oap
{

class Network
{
public: // types

  class IController
  {
    public:
      virtual ~IController() {}
      virtual bool shouldCalculateError(uintt step) = 0;
      virtual void setError (floatt value, oap::ErrorType type) = 0;
      virtual bool shouldContinue() = 0;
  };

public:
  using Matrices = std::vector<math::ComplexMatrix*>;

  Network (oap::generic::SingleMatrixProcedures* smp, oap::generic::MultiMatricesProcedures* mmp, oap::NetworkGenericApi* nga, bool deallocate);
  virtual ~Network();

  Network (const Network&) = delete;
  Network (Network&&) = delete;
  Network& operator= (const Network&) = delete;
  Network& operator= (Network&&) = delete;

  /**
   *  Initializes weights by random values
   */
  void initWeights (bool init);

  void initTopology (const std::vector<uintt>& topology, const std::vector<uintt>& biases, const std::vector<Activation>& activations);
  void initInput (const oap::InputTopology& itopology);

  oap::Layer* createLayer (uintt neurons, const Activation& activation = Activation::SIGMOID, LayerType layerType = LayerType::ONE_MATRIX);
  oap::Layer* createLayer (uintt neurons, bool addBias, const Activation& activation = Activation::SIGMOID, LayerType layerType = LayerType::ONE_MATRIX);
  void createLevel (oap::Layer* layer, LayerType layerType = LayerType::ONE_MATRIX);

  LHandler createFPLayer (uintt samples, LayerType ltype = LayerType::ONE_MATRIX);
  LHandler createSharedFPLayer (const std::vector<LHandler>& handlers, LayerType ltype = LayerType::ONE_MATRIX);
  LHandler createSharedFPLayer (const std::vector<std::vector<FPMatrices*>>& fpmatrices, LayerType ltype = LayerType::ONE_MATRIX);

  oap::HostComplexMatrixUPtr run (const math::ComplexMatrix* hostInputs, ArgType argType, oap::ErrorType errorType);
  void setHostInputs (math::ComplexMatrix* inputs, uintt index);

  void setInputs (math::ComplexMatrix* inputs, ArgType argType, LHandler handler = 0);
  void setInputs (const Matrices& inputs, ArgType argType, LHandler handler = 0);

  void destroyFPSection (LHandler handle);

  math::ComplexMatrix* getOutputs (math::ComplexMatrix* outputs, ArgType argType, LHandler handler = 0) const;
  void getOutputs (Matrices& outputs, ArgType argType, LHandler handler = 0) const;

  math::ComplexMatrix* getHostOutputs () const;

  math::MatrixInfo getOutputInfo () const;
  math::MatrixInfo getInputInfo () const;

  void forwardPropagation (LHandler handler = 0);
  void fbPropagation (LHandler handler, oap::ErrorType errorType, CalculationType calcType);
  void accumulateErrors (oap::ErrorType errorType, CalculationType calcType, LHandler handler = 0);

  math::ComplexMatrix* getErrors (ArgType type) const;

  floatt calculateMSE ();
  floatt calculateRMSE ();
  floatt calculateSum ();
  floatt calculateSumMean ();
  floatt calculateCrossEntropy ();

  floatt calculateError (oap::ErrorType errorType);

  void backPropagation (LHandler handler = 0);

  void updateWeights (LHandler handler = 0);

  bool train (const Matrices& hostInputs, const Matrices& expectedHostOutputs, ArgType argType, oap::ErrorType errorType);
  bool train (math::ComplexMatrix* hostInputs, math::ComplexMatrix* expectedHostOutputs, ArgType argType, oap::ErrorType errorType);

  void setController (IController* icontroller);

  void setHostWeights (math::ComplexMatrix* weights, uintt layerIndex);

  void getHostWeights (math::ComplexMatrix* weights, uintt layerIndex);

  void setDeviceWeights (math::ComplexMatrix* weights, uintt layerIndex);

  void setLearningRate (floatt lr);
  floatt getLearningRate () const;

  uintt getLayersCount () const
  {
    return m_networkTopology.size ();
  }

  uintt getNeuronsCount (uintt layerIdx) const;
  uintt getBiasesCount (uintt layerIdx) const;
  Activation getActivation (uintt layerIdx) const;
  BPMatrices* getBPMatrices (uintt layerIdx) const;
  NBPair getNBPair(uintt layerIdx) const;

  Layer* getLayer(uintt layerIndex, LHandler handler = 0) const;

  bool operator== (const Network& network) const;
  bool operator!= (const Network& network) const;

  void printLayersWeights () const;


  template<typename Callback>
  void iterateLayers (Callback&& callback)
  {
    for (auto it = m_layers.cbegin(); it != m_layers.cend(); ++it)
    {
      callback (*it);
    }
  }

  void setHostInputs (const Matrices& inputs, uintt layerIndex);

  void setExpected (math::ComplexMatrix* expected, ArgType argType, LHandler handler = 0);
  void setExpected (const std::vector<math::ComplexMatrix*>& expected, ArgType argType, LHandler handler = 0);
  Matrices getExpected (LHandler handler) const;
  void printLayersInputs () const;

  void postStep (Layer* layer);
  void postStep ();

  void resetErrors (Layer* layer);
  void resetErrors ();
  void resetErrorsVec ();

private:
  NetworkGenericApi* m_nga;

  struct GenericFPLayerArgs
  {
    uintt samples;
    std::vector<std::vector<FPMatrices*>> fpmatrices;
  };

  LHandler createGenericFPLayer (LayerType ltype, const GenericFPLayerArgs& args);

  void addLayer (Layer* layer, LayerType layerType = LayerType::ONE_MATRIX);
  void addToTopology(Layer* layer);
  void addToTopologyBPMatrices(Layer* layer);

  std::vector<floatt> m_errorsVec;

  floatt m_learningRate = 0.1f;
  uintt m_step = 1;

  using ExpectedOutputs = std::map<LHandler, Matrices>;
  ExpectedOutputs m_expectedOutputs;

  using Layers = std::vector<Layer*>;
  using LayersVec = std::vector<Layers>;

  LayersVec m_layers;
  std::map<LHandler, LayerType> m_layerType;
  LayerType getType (LHandler handler) const
  {
    return m_layerType.at(handler);
  }

  std::vector<FPMatrices*> m_AllFpMatricesVec;
  std::vector<BPMatrices*> m_AllBpMatricesVec;

  void deallocateFPMatrices();
  void deallocateBPMatrices();

  void destroyNetwork();
  void destroyLayers();

  void destroyInnerExpectedMatrices();
  bool m_innerExpectedMatrices = false;

  inline void calculateError();

  bool shouldContinue (oap::ErrorType errorType);

  oap::generic::SingleMatrixProcedures* m_singleApi;
  oap::generic::MultiMatricesProcedures* m_multiApi;
  bool m_deallocate;
  IController* m_icontroller = nullptr;

  std::ostream& log()
  {
    //std::cout << "[network] ";
    return std::cout;
  }

  std::vector<uintt> m_networkTopology;
  std::vector<uintt> m_networkBiases;
  std::vector<Activation> m_networkActivations;
  std::vector<BPMatrices*> m_bpMatricesNetwork;
  bool m_isCreatedByNetworkTopology = false;
  bool m_isCreatedByApi = false;
  bool m_initWeights = true;

  std::map<Layer*, std::vector<math::Matrix*>> m_errMatrices;

  template<typename LayerT, typename AllocNeuronsApi>
  friend void allocateNeurons (LayerT& ls, uintt neuronsCount, uintt biasCount);

  void setHostInputs (Layer* layer, const Matrices& inputs);
  void setDeviceInputs (Layer* layer, const Matrices& inputs);
  void setHostInputs (Layer* layer, const math::ComplexMatrix* inputs);
  void setDeviceInputs (Layer* layer, const math::ComplexMatrix* inputs);
  void setDeviceWeights (Layer* layer, const math::ComplexMatrix* weights);

  void setExpectedProtected (typename ExpectedOutputs::mapped_type& holder, const std::vector<math::ComplexMatrix*>& expected, ArgType argType);
};
}
#endif
