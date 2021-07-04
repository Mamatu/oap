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

#ifndef OAP_GENERIC_NEURAL_UTILS_H
#define OAP_GENERIC_NEURAL_UTILS_H

#include <algorithm>
#include <iterator>
#include <random>

#include "Math.hpp"
#include "Logger.hpp"

#include "oapLayerStructure.hpp"
#include "oapRandomGenerator.hpp"
#include "MatrixAPI.hpp"

#include "oapHostComplexMatrixUPtr.hpp"
#include "oapMatrixRandomGenerator.hpp"
#include "oapGenericNeuralApi.hpp"

namespace oap
{
namespace nutils
{

template<typename Container, typename RandomFunc>
Container splitIntoTestAndTrainingSet (Container& trainingSet, Container& testSet, const Container& data, size_t trainingSize, size_t testSize, RandomFunc&& rf)
{
  debugAssert (data.size() == trainingSize + testSize);

  Container modifiableData = data;
  std::random_shuffle (modifiableData.begin(), modifiableData.end(), rf);

  trainingSet.resize (trainingSize);
  testSet.resize (modifiableData.size() - trainingSize);

  auto copyIt = modifiableData.begin();
  std::advance (copyIt, trainingSet.size());

  std::copy(modifiableData.begin(), copyIt, trainingSet.begin());
  std::copy(copyIt, modifiableData.end(), testSet.begin());

  logInfo ("training set: %lu", trainingSet.size());
  logInfo ("test set: %lu", testSet.size());

  return modifiableData;
}

template<typename Container>
Container splitIntoTestAndTrainingSet (Container& trainingSet, Container& testSet, const Container& data, size_t trainingSize, size_t testSize, oap::utils::RandomGenerator& rg)
{
  auto randomFunc = [&rg](int n)
  {
    int i = static_cast<int>(rg(0, static_cast<floatt>(n)));
    logTrace ("randomFunc = %d\n",i);
    return i;
  };

  return splitIntoTestAndTrainingSet (trainingSet, testSet, data, trainingSize, testSize, randomFunc);
}

template<typename Array>
floatt mean (Array&& array)
{
  floatt sum = 0;
  for (floatt v : array)
  {
    sum += v;
  }
  return sum / array.size();
}

template<typename Array>
floatt std (Array&& array, floatt mean)
{
  floatt sd = 0;
  for (floatt v : array)
  {
    floatt r = v - mean;
    sd += r * r;
  }
  sd = sqrt (sd / (array.size() - 1));
  return sd;
}

template<typename Array>
floatt std (Array&& array)
{
  floatt mean = mean (array);
  return std (array, mean);
}

template<typename Container>
void scale (Container&& container)
{
  floatt mn = mean (container);
  floatt sd = std (container, mn);
  for (floatt& v : container)
  {
    v = (v - mn) / sd;
  }
}

template<typename T>
void scale (T* container, size_t length)
{
  floatt mn = mean (container);
  floatt sd = std (container, mn);
  for (size_t idx = 0; idx < length; ++idx)
  {
    floatt v = container [idx];
    v = (v - mn) / sd;
    container [idx] = v;
  }
}

template<typename Container>
Container scale (const Container& container)
{
  Container ncontainer = container;
  scale (std::move (ncontainer));
  return ncontainer;
}

template<typename Container>
Container splitIntoTestAndTrainingSet (Container& trainingSet, Container& testSet, const Container& data, floatt rate, oap::utils::RandomGenerator& rg)
{
  debugAssert (rate > 0 && rate <= 1);

  const size_t trainingSize = rate * data.size();

  return splitIntoTestAndTrainingSet (trainingSet, testSet, data, trainingSize, data.size() - trainingSize, rg);
}

template<typename Container, typename Callback>
void iterate (const Container& container, Callback&& callback)
{
  debugAssert (container.size() > 0);
  for (size_t idx = 0; idx < container.size(); ++idx)
  {
    callback (container, idx);
  }
}

template<typename Container, typename GetSize>
size_t getElementsCount (const Container& container, GetSize&& getSize)
{
  size_t count = 0;
  for (size_t idx = 0; idx < container.size(); ++idx)
  {
    count += getSize (container[idx]);
  }
  return count;
}

template<typename Container>
size_t getElementsCount (const Container& container)
{
  size_t count = 0;
  for (size_t idx = 0; idx < container.size(); ++idx)
  {
    count += container[idx].size();
  }
  return count;
}

template<typename Network, typename Container, typename CreateMatrix, typename CopyBufferToMatrix>
void createExpectedOutput (Network* network, LHandler handler, const Container& container2D, ArgType argType, CreateMatrix&& createMatrix, CopyBufferToMatrix&& copyBufferToMatrix)
{
  std::vector<math::ComplexMatrix*> matrices;
  for (uintt idx = 0; idx < container2D.size(); ++idx)
  {
    matrices.push_back (createMatrix (1, container2D[idx].size()));
  }

  iterate (container2D, [&matrices, &copyBufferToMatrix](const Container& container2D, size_t idx)
  {
    for (uintt idx = 0; idx < matrices.size(); ++idx)
    {
      const size_t size = container2D[idx].size();
      copyBufferToMatrix (matrices[idx], container2D[idx].data(), size);
    }
  });

  network->setExpected (matrices, argType, handler); // std::move
}

template<template<typename, typename> class Vec, typename GetMatrixInfo, typename CopyMatrixToBuffer>
Vec<floatt, std::allocator<floatt>> convertToFloattBuffer (const Vec<math::ComplexMatrix*, std::allocator<math::ComplexMatrix*>>& matrices, GetMatrixInfo&& getMatrixInfo, CopyMatrixToBuffer&& copyMatrixToBuffer)
{
  uintt length = getElementsCount (matrices, [&getMatrixInfo](const math::ComplexMatrix* matrix)
      {
        math::MatrixInfo minfo = getMatrixInfo(matrix);
        return minfo.columns() * minfo.rows();
      });

  Vec<floatt, std::allocator<floatt>> buffer;
  buffer.resize (length);
  uintt pos = 0;

  for (uintt idx = 0; idx < matrices.size(); ++idx)
  {
    math::MatrixInfo matrixInfo = getMatrixInfo(matrices[idx]);
    uintt sublength = matrixInfo.columns() * matrixInfo.rows();
    copyMatrixToBuffer (&buffer[pos], sublength, matrices[idx]);
    pos += sublength;
  }
  return buffer;
}

namespace
{
template<typename Container>
class VectorHandler
{
  std::vector<floatt>& m_vec;
  const Container& m_container;
  uintt m_idx;
  public:
    VectorHandler (std::vector<floatt>& vec, const Container& container, uintt idx) : m_vec(vec), m_container (container), m_idx(idx)
    {}

    void call()
    {
      for (uintt idx1 = 0; idx1 < m_container[m_idx].size(); ++idx1)
      {
        m_vec.push_back (m_container[m_idx][idx1]);
      }
    }
};

template<typename Container>
class FloattHandler
{
  std::vector<floatt>& m_vec;
  const Container& m_container;
  uintt m_idx;
  public:
    FloattHandler (std::vector<floatt>& vec, const Container& container, uintt idx) : m_vec(vec), m_container (container), m_idx(idx)
    {}

    void call()
    {
      for (uintt idx1 = 0; idx1 < m_container[m_idx].size(); ++idx1)
      {
        m_vec.push_back (m_container[m_idx]);
      }
    }
};

template<typename Container>
class NotSupportedHandler
{
  public:
    NotSupportedHandler (std::vector<floatt>& vec, const Container& container, uintt idx)
    {}

    void call()
    {
      oapAssert ("Not supported type");
    }
};
}

template<typename LayerT, typename CopyBufferToMatrix>
void copyToInputs_multiMatrices (LayerT* ilayer, size_t index, const floatt* buffer, size_t size, CopyBufferToMatrix&& copyBufferToMatrix)
{
  copyBufferToMatrix (ilayer->getFPMatrices(index)->m_inputs, buffer, size);
}

template<typename LayerT, typename Container, typename CopyBufferToMatrix>
void copyToInputs_multiMatrices (LayerT* ilayer, const Container& container2D, CopyBufferToMatrix&& copyBufferToMatrix)
{
  size_t fsize = 0;
  debugAssert (container2D.size() > 0);
  iterate (container2D, [ilayer, &copyBufferToMatrix, &fsize](const Container& container2D, size_t idx)
  {
    if (idx == 0) { fsize = container2D[idx].size(); }
    debugAssert (fsize == container2D[idx].size());
    copyToInputs (ilayer, idx, container2D[idx].data(), container2D[idx].size(), copyBufferToMatrix);
  });
}

template<typename LayerT, typename CopyBufferToMatrix>
void copyToInputs_oneMatrix (LayerT* ilayer, const floatt* buffer, size_t size, CopyBufferToMatrix&& copyBufferToMatrix)
{
  copyBufferToMatrix (ilayer->getFPMatrices()->m_inputs, buffer, size);
}

template<typename LayerT, typename Container, typename CopyBufferToMatrix>
void copyToInputs_oneMatrix (LayerT* ilayer, const Container& container, CopyBufferToMatrix&& copyBufferToMatrix)
{
  uintt length = oap::nutils::getElementsCount (container);
  std::vector<floatt> buffer;
  buffer.reserve(length);
  size_t fsize = 0;
  iterate (container, [ilayer, &buffer, &copyBufferToMatrix, &fsize](const Container& container2D, size_t idx)
  {
    constexpr bool isVector = std::is_same<std::vector<floatt>, typename Container::value_type>::value;
    constexpr bool isFloatt = std::is_same<floatt, typename Container::value_type>::value;
    typename std::conditional<isVector, VectorHandler<Container>, typename std::conditional<isFloatt, FloattHandler<Container>, NotSupportedHandler<Container>>::type>::type
      obj(buffer, container2D, idx);

    obj.call();
  });
  copyToInputs_oneMatrix (ilayer, buffer.data(), buffer.size(), copyBufferToMatrix);
}

template<typename NetworkT, typename Callback>
void iterateNetwork (NetworkT& network, Callback&& callback)
{
  for (size_t idx = 0; idx < network.getLayersCount() - 1; ++idx)
  {
    auto* clayer = network.getLayer (idx);
    auto* nlayer = network.getLayer (idx + 1);

    callback (*clayer, *nlayer);
  }
}

template<typename LayerT>
class BiasesFilter final
{
  public:
    template<typename GetWeightsInfo>
    BiasesFilter (const LayerT& currentLayerT, const LayerT& nextLayerT, GetWeightsInfo&& getWeightsInfo) :
      BiasesFilter (getWeightsInfo (currentLayerT), nextLayerT)
    {}

    BiasesFilter (const math::MatrixInfo& layerWeightsInfo, const LayerT& nextLayerT) :
      m_winfo (layerWeightsInfo), m_nextLayerT (nextLayerT)
    {}

    floatt operator()(uintt c, uintt r, floatt v) const
    {
      if (m_nextLayerT.getBiasesCount() == 1 && m_winfo.rows() - 1 == r)
      {
        return 0.;
      }
      return v;
    }

  private:
    math::MatrixInfo m_winfo;
    const LayerT& m_nextLayerT;
};

template<typename LayerT, typename MatrixRandomGenerator>
oap::HostComplexMatrixUPtr createRandomMatrix (LayerT& layer, const math::MatrixInfo& minfo, MatrixRandomGenerator&& mrg)
{
  oap::HostComplexMatrixUPtr randomMatrix = oap::chost::NewReMatrix (minfo.columns(), minfo.rows());

  for (uintt c = 0; c < minfo.columns(); ++c)
  {
    for (uintt r = 0; r < minfo.rows(); ++r)
    {
      SetRe (randomMatrix.get(), c, r, mrg(c, r));
    }
  }

  //rg (randomMatrix.get(), ArgType::HOST);

  return randomMatrix;
}

template<typename LayerT, typename GetMatrixInfo, typename CopyHostMatrixToKernelMatrix, typename MatrixRandomGenerator>
void initRandomWeights (LayerT& layer, const LayerT& nextLayer, GetMatrixInfo&& getMatrixInfo, CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix, MatrixRandomGenerator&& mrg)
{
  math::MatrixInfo winfo = oap::generic::getWeightsInfo (layer, getMatrixInfo);

  auto randomMatrix = createRandomMatrix (layer, winfo, mrg);

  oap::generic::setWeights (layer, randomMatrix.get (), copyHostMatrixToKernelMatrix);
}

template<typename LayerT, typename GetMatrixInfo, typename CopyHostMatrixToKernelMatrix, typename Range = std::pair<floatt, floatt>>
void initRandomWeightsByRange (LayerT& layer, const LayerT& nextLayer, GetMatrixInfo&& getMatrixInfo, CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix, Range&& range = std::pair<floatt, floatt>(-0.5, 0.5))
{
  math::MatrixInfo winfo = oap::generic::getWeightsInfo (layer, getMatrixInfo);

  oap::utils::MatrixRandomGenerator rg (range.first, range.second);
  rg.setFilter (oap::nutils::BiasesFilter<LayerT> (winfo, nextLayer));

  auto randomMatrix = createRandomMatrix (layer, winfo, rg);

  oap::generic::setWeights (layer, randomMatrix.get (), copyHostMatrixToKernelMatrix);
  //rg (getWeights (layer), ArgType::DEVICE);
}

}
}

#endif
