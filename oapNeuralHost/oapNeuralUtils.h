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

#ifndef OAP_NEURAL_UTILS_H
#define OAP_NEURAL_UTILS_H

#include <algorithm>
#include <iterator>
#include <random>

#include "Math.h"
#include "Logger.h"

namespace oap
{
namespace nutils
{
template<typename Container>
Container splitIntoTestAndTrainingSet (Container& trainingSet, Container& testSet, const Container& data, floatt rate)
{
  debugAssert (rate > 0 && rate <= 1);

  Container modifiableData = data;

  std::random_shuffle (modifiableData.begin(), modifiableData.end());
  size_t trainingSetLength = modifiableData.size() * rate;

  trainingSet.resize (trainingSetLength);
  testSet.resize (modifiableData.size() - trainingSetLength);

  auto copyIt = modifiableData.begin();
  std::advance (copyIt, trainingSet.size());

  std::copy(modifiableData.begin(), copyIt, trainingSet.begin());
  std::copy(copyIt, modifiableData.end(), testSet.begin());

  logInfo ("training set: %lu", trainingSet.size());
  logInfo ("test set: %lu", testSet.size());

  return modifiableData;
};
}
}

#endif
