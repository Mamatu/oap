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

#ifndef OAP_NEURAL_NETWORK_H
#define OAP_NEURAL_NETWORK_H

#include "NeuralTypes.h"

#include "oapLayer.h"
#include "oapDeviceMatrixPtr.h"

#include "oapNetworkStructure.h"

class Network : private NetworkS
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

  Network();
  virtual ~Network();

  Network (const Network&) = delete;
  Network (Network&&) = delete;
  Network& operator= (const Network&) = delete;
  Network& operator= (Network&&) = delete;

  Layer* createLayer (uintt neurons, const Activation& activation = Activation::SIGMOID);
  Layer* createLayer (uintt neurons, bool addBias, const Activation& activation = Activation::SIGMOID);

  oap::HostMatrixUPtr run (math::Matrix* hostInputs, ArgType argType, oap::ErrorType errorType);

  void setInputs (math::Matrix* inputs, ArgType argType, FPHandler handler = 0);
  void setExpected (math::Matrix* expected, ArgType argType, FPHandler handler = 0);

  FPHandler createFPSection (uintt samples);
  void destroyFPSection (FPHandler handle);

  math::Matrix* getOutputs (math::Matrix* outputs, ArgType argType, FPHandler handler = 0) const;
  math::Matrix* getHostOutputs () const;

  math::MatrixInfo getOutputInfo () const;
  math::MatrixInfo getInputInfo () const;

  void forwardPropagation (FPHandler handler = 0);
  void accumulateErrors (oap::ErrorType errorType, CalculationType calcType, FPHandler handler = 0);

  math::Matrix* getErrors (ArgType type) const;

  floatt calculateMSE ();
  floatt calculateRMSE ();
  floatt calculateSum ();
  floatt calculateSumMean ();
  floatt calculateCrossEntropy ();

  floatt calculateError (oap::ErrorType errorType);

  void backPropagation ();

  void updateWeights();

  bool train (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs, ArgType argType, oap::ErrorType errorType);

  void createLevel (Layer* layer);

  void addLayer (Layer* layer);

  void setController (IController* icontroller);

  void setHostWeights (math::Matrix* weights, uintt layerIndex);

  void getHostWeights (math::Matrix* weights, uintt layerIndex);

  void setDeviceWeights (math::Matrix* weights, uintt layerIndex);

  void setLearningRate (floatt lr);
  floatt getLearningRate () const;

  void save (utils::ByteBuffer& buffer) const;

  static Network* load (const utils::ByteBuffer& buffer);

  uintt getLayersCount () const
  {
    return m_layers.size ();
  }
  
  Layer* getLayer(uintt layerIndex) const;

  bool operator== (const Network& network) const;
  bool operator!= (const Network& network) const;

  void printLayersWeights ();

  void postStep (LayerS* layer);
  void postStep ();

  void resetErrors (LayerS* layer);
  void resetErrors ();
  void resetErrorsVec ();

  template<typename Callback>
  void iterateLayers (Callback&& callback)
  {
    for (auto it = m_layers.cbegin(); it != m_layers.cend(); ++it)
    {
      callback (*it);
    }
  }

  LayerS_FP* getLayerS_FP (FPHandler handler, size_t idx) const
  {
    return m_layers[idx]->getLayerS_FP (handler);
  }

protected:
  void setHostInputs (math::Matrix* inputs, uintt layerIndex);

private:
  std::vector<Layer*> m_layers;

  void destroyNetwork();
  void destroyLayers();
  void destroyFPSections();
  void destroyFPSection (LayerS_FP*);

  inline void calculateError();

  bool shouldContinue (oap::ErrorType errorType);

  oap::CuProceduresApi m_cuApi;

  std::map<FPHandler, oap::DeviceMatrixPtr> m_expectedDeviceOutputs;
  IController* m_icontroller = nullptr;

  std::ostream& log()
  {
    //std::cout << "[network] ";
    return std::cout;
  }

  template<typename LayerT, typename AllocNeuronsApi>
  friend void allocateNeurons (LayerT& ls, uintt neuronsCount, uintt biasCount);
};

#endif
