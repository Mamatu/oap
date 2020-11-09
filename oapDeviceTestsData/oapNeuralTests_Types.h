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

#ifndef OAP_NEURAL_TESTS__TYPES_H
#define OAP_NEURAL_TESTS__TYPES_H

#include <vector>
#include <utility>
#include <tuple>

#include "Math.h"
#include "oapLayerStructure.h"

using Weights = std::vector<floatt>;
using WeightsInSteps = std::vector<Weights>;
using WeightsLayers = std::vector<WeightsInSteps>;

using Point = std::pair<floatt, floatt>;
using PointLabel = std::pair<Point, floatt>;
using Points = std::vector<PointLabel>;

using Batch = Points;
using Batches = std::vector<Batch>;
using BatchesLHandlers = std::vector<LHandler>;

using PointsLoss = std::pair<Points, floatt>;

using Step = std::tuple<Batches, PointsLoss, PointsLoss, BatchesLHandlers>;
using Steps = std::vector<Step>;

using IdxsToCheck = std::vector<std::vector<size_t>>;

inline Step createStep (const Batches& batches, const Points& points, floatt error, const Points& points1, floatt error1)
{
  return std::make_tuple(batches, std::make_pair(points, error), std::make_pair(points1, error1), BatchesLHandlers());
}

inline Step createStep (const Batches& batches)
{
  return std::make_tuple(batches, std::make_pair(Points(), -1), std::make_pair(Points(), -1), BatchesLHandlers());
}

inline const Batches& getBatches (const Step& step)
{
  return std::get<0>(step);
}

inline const Points& getFront (const Batches& batches)
{
  return batches[0];
}

inline const Points& getFront (const Steps& steps)
{
  return getFront (std::get<0>(steps[0]));
}

inline size_t getBatchesCount (const Steps& steps)
{
  return std::get<0>(steps[0]).size();
}

inline size_t getFBSize (const Steps& steps)
{
  return getFront (steps).size();
}

inline const Points& getBatch (const Batches& batches, size_t idx)
{
  return batches[idx];
}

#endif
