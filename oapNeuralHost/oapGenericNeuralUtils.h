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

#include "Math.h"
#include "Logger.h"

#include "oapLayerStructure.h"
#include "MatrixAPI.h"

namespace oap
{
namespace nutils
{

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

template<typename Network, typename Container2D, typename CreateMatrix, typename CopyBufferToMatrix>
void createExpectedOutput (Network* network, LHandler handler, const Container2D& container2D, ArgType argType, CreateMatrix&& createMatrix, CopyBufferToMatrix&& copyBufferToMatrix)
{
  std::vector<math::Matrix*> matrices;
  for (uintt idx = 0; idx < container2D.size(); ++idx)
  {
    matrices.push_back (createMatrix (1, container2D[idx].size()));
  }

  iterate (container2D, [&matrices, &copyBufferToMatrix](const Container2D& container2D, size_t idx)
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
Vec<floatt, std::allocator<floatt>> convertToFloattBuffer (const Vec<math::Matrix*, std::allocator<math::Matrix*>>& matrices, GetMatrixInfo&& getMatrixInfo, CopyMatrixToBuffer&& copyMatrixToBuffer)
{
  uintt length = getElementsCount (matrices, [&getMatrixInfo](const math::Matrix* matrix)
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

template<typename LayerT, typename CopyBufferToMatrix>
void copyToInputs_multiMatrices (LayerT* ilayer, size_t index, const floatt* buffer, size_t size, CopyBufferToMatrix&& copyBufferToMatrix)
{
  copyBufferToMatrix (ilayer->getFPMatrices(index)->m_inputs, buffer, size);
}

template<typename LayerT, typename Container2D, typename CopyBufferToMatrix>
void copyToInputs_multiMatrices (LayerT* ilayer, const Container2D& container2D, CopyBufferToMatrix&& copyBufferToMatrix)
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

template<typename LayerT, typename CopyBufferToMatrix>
void copyToInputs_oneMatrix (LayerT* ilayer, const floatt* buffer, size_t size, CopyBufferToMatrix&& copyBufferToMatrix)
{
  copyBufferToMatrix (ilayer->getFPMatrices()->m_inputs, buffer, size);
}

namespace
{
template<typename Container2D>
class VectorHandler
{
  std::vector<floatt>& m_vec;
  const Container2D& m_container;
  uintt m_idx;
  public:
    VectorHandler (std::vector<floatt>& vec, const Container2D& container, uintt idx) : m_vec(vec), m_container (container), m_idx(idx)
    {}

    void call()
    {
      for (uintt idx1 = 0; idx1 < m_container[m_idx].size(); ++idx1)
      {
        m_vec.push_back (m_container[m_idx][idx1]);
      }
    }
};

template<typename Container2D>
class FloattHandler
{
  std::vector<floatt>& m_vec;
  const Container2D& m_container;
  uintt m_idx;
  public:
    FloattHandler (std::vector<floatt>& vec, const Container2D& container, uintt idx) : m_vec(vec), m_container (container), m_idx(idx)
    {}

    void call()
    {
      for (uintt idx1 = 0; idx1 < m_container[m_idx].size(); ++idx1)
      {
        m_vec.push_back (m_container[m_idx]);
      }
    }
};

template<typename Container2D>
class NotSupportedHandler
{
  public:
    NotSupportedHandler (std::vector<floatt>& vec, const Container2D& container, uintt idx)
    {}

    void call()
    {
      oapAssert ("Not supported type");
    }
};
}

template<typename LayerT, typename Container2D, typename CopyBufferToMatrix>
void copyToInputs_oneMatrix (LayerT* ilayer, const Container2D& container2D, CopyBufferToMatrix&& copyBufferToMatrix)
{
  uintt length = oap::nutils::getElementsCount (container2D);
  std::vector<floatt> buffer;
  buffer.reserve(length);
  size_t fsize = 0;
  iterate (container2D, [ilayer, &buffer, &copyBufferToMatrix, &fsize](const Container2D& container2D, size_t idx)
  {
    constexpr bool isVector = std::is_same<std::vector<floatt>, typename Container2D::value_type>::value;
    constexpr bool isFloatt = std::is_same<floatt, typename Container2D::value_type>::value;
    typename std::conditional<isVector, VectorHandler<Container2D>, typename std::conditional<isFloatt, FloattHandler<Container2D>, NotSupportedHandler<Container2D>>::type>::type
      obj(buffer, container2D, idx);

    obj.call();
  });
  copyToInputs_oneMatrix (ilayer, buffer.data(), buffer.size(), copyBufferToMatrix);
}

}
}

#endif
