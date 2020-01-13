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

#ifndef OAP_GENERIC_NEURAL_UTILS_H
#define OAP_GENERIC_NEURAL_UTILS_H

#include <algorithm>
#include <iterator>
#include <random>

#include "Math.h"
#include "Logger.h"

#include "oapLayerStructure.h"
#include "oapNetworkStructure.h"

namespace oap
{
namespace nutils
{

inline void copyHostBufferToHostReMatrix (math::Matrix* matrix, size_t index, const floatt* buffer, size_t size)
{
  floatt* re = matrix->reValues;
  re += index * size;
  memcpy (re, buffer, size * sizeof(floatt));
}

template<typename Container>
Container splitIntoTestAndTrainingSet (Container& trainingSet, Container& testSet, const Container& data, size_t trainingSize, size_t testSize)
{
  debugAssert (data.size() == trainingSize + testSize);

  Container modifiableData = data;
  std::random_shuffle (modifiableData.begin(), modifiableData.end());

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
Container splitIntoTestAndTrainingSet (Container& trainingSet, Container& testSet, const Container& data, floatt rate)
{
  debugAssert (rate > 0 && rate <= 1);

  const size_t trainingSize = rate * data.size();

  return splitIntoTestAndTrainingSet (trainingSet, testSet, data, trainingSize, data.size() - trainingSize);
}

template<typename CopyBufferToMatrix>
void copyTo (math::Matrix* matrix, size_t index, const floatt* buffer, size_t size, CopyBufferToMatrix&& copyBufferToMatrix)
{
  copyBufferToMatrix (matrix, index, buffer, size);
}

template<typename LayerT, typename CopyBufferToMatrix>
void copyToInputs (LayerT* ilayer, size_t index, const floatt* buffer, size_t size, CopyBufferToMatrix&& copyBufferToMatrix)
{
  copyTo (ilayer->getFPMatrices()->m_inputs, index, buffer, size, copyBufferToMatrix);
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

template<typename Container>
size_t getElementsCount (const Container& container)
{
  size_t count = 0;
  for (size_t idx = 0; idx < container.size(); ++idx)
  {
    count += container[idx].size ();
  }
  return count;
}

template<typename LayerT, typename Container2D, typename CopyBufferToMatrix>
void copyToInputs (LayerT* ilayer, const Container2D& container2D, CopyBufferToMatrix&& copyBufferToMatrix)
{
  size_t fsize = 0;
  debugAssert (container2D.size() > 0);
  iterate (container2D, [ilayer, &copyBufferToMatrix, &fsize](const Container2D& container2D, size_t idx)
  {
    if (idx == 0) { fsize = container2D[idx].size(); }
    debugAssert (fsize == container2D[idx].size());
    copyToInputs (ilayer, idx, container2D[idx].data(), container2D[idx].size(), copyBufferToMatrix);
  });
}

template<typename T, typename Container2D, typename CreateMatrix, typename CopyBufferToMatrix>
void createExpectedOutput (NetworkS<T>* network, FPHandler handler, Container2D container2D, ArgType argType, CreateMatrix&& createMatrix, CopyBufferToMatrix&& copyBufferToMatrix)
{
  math::Matrix* matrix = network->getExpected (handler);

  if (matrix == nullptr)
  {
    size_t containerLength = oap::nutils::getElementsCount (container2D);

    network->setExpected (createMatrix (1, containerLength), argType, handler);
    matrix = network->getExpected (handler);
  }

  iterate (container2D, [&matrix, &copyBufferToMatrix](const Container2D& container2D, size_t idx)
  {
    const size_t size = container2D[idx].size();
    oap::nutils::copyTo (matrix, idx, container2D[idx].data(), size, copyBufferToMatrix);
  });
}

}
}

#endif
