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

#include "oapDeviceLayer.h"
#include "CuProceduresApi.h"
#include "oapDeviceMatrixPtr.h"

#include "oapNetworkStructure.h"

class Network : public NetworkS<oap::DeviceMatrixPtr>
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

  Network(oap::CuProceduresApi* calcApi = nullptr);
  virtual ~Network();

  Network (const Network&) = delete;
  Network (Network&&) = delete;
  Network& operator= (const Network&) = delete;
  Network& operator= (Network&&) = delete;

  DeviceLayer* createLayer (uintt neurons, const Activation& activation = Activation::SIGMOID, bool binitWeights = true);
  DeviceLayer* createLayer (uintt neurons, bool addBias, const Activation& activation = Activation::SIGMOID, bool binitWeights = true);

  void createLevel (DeviceLayer* layer, bool binitWeights = true);

  void addLayer (DeviceLayer* layer);

  FPHandler createFPLayer (uintt samples);

  oap::HostMatrixUPtr run (const math::Matrix* hostInputs, ArgType argType, oap::ErrorType errorType);

  void setInputs (const math::Matrix* inputs, ArgType argType, FPHandler handler = 0);

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

  void backPropagation (FPHandler handler = 0);

  void updateWeights (FPHandler handler = 0);

  bool train (math::Matrix* hostInputs, math::Matrix* expectedHostOutputs, ArgType argType, oap::ErrorType errorType);


  void setController (IController* icontroller);

  void setHostWeights (math::Matrix* weights, uintt layerIndex);

  void getHostWeights (math::Matrix* weights, uintt layerIndex);

  void setDeviceWeights (math::Matrix* weights, uintt layerIndex);

  void setLearningRate (floatt lr);
  floatt getLearningRate () const;

  //void save (utils::ByteBuffer& buffer) const;

  //static Network* load (const utils::ByteBuffer& buffer);

  uintt getLayersCount () const
  {
    return m_layers[0].size ();
  }
  
  DeviceLayer* getLayer(uintt layerIndex, FPHandler handler = 0) const;

  bool operator== (const Network& network) const;
  bool operator!= (const Network& network) const;

  void printLayersWeights () const;

  void postStep (DeviceLayer* layer);
  void postStep ();

  void resetErrors (DeviceLayer* layer);
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

protected:
  void setHostInputs (math::Matrix* inputs, uintt layerIndex);
  virtual void setExpectedProtected (typename ExpectedOutputs::mapped_type& holder, math::Matrix* expected, ArgType argType) override;
  virtual math::Matrix* convertExpectedProtected (oap::DeviceMatrixPtr t) const override;

private:

  using Layers = std::vector<DeviceLayer*>;
  using LayersVec = std::vector<Layers>;

  LayersVec m_layers;

  std::vector<FPMatrices*> m_fpMatricesVec;
  std::vector<BPMatrices*> m_bpMatricesVec;

  void deallocateFPMatrices();
  void deallocateBPMatrices();

  void destroyNetwork();
  void destroyLayers();

  inline void calculateError();

  bool shouldContinue (oap::ErrorType errorType);

  oap::CuProceduresApi* m_cuApi;
  bool m_releaseCuApi;
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
