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

#ifndef OAP_POINTS_CLASSIFICATION__TEST_H
#define OAP_POINTS_CLASSIFICATION__TEST_H

#include "Math.hpp"
#include "oapProcedures.hpp"
#include "oapNetworkGenericApi.hpp"

namespace oap
{
class Coordinate
{
  public:
    floatt q;
    floatt r;
    floatt label;

    Coordinate()
    {
      q = 0;
      r = 0;
      label = 0;
    }

    Coordinate(floatt _q, floatt _r, floatt _label)
    {
      q = _q;
      r = _r;
      label = _label;
    }

    void fitRadius (floatt min, floatt max)
    {
      r = (r - min) / (max - min);
    }

    bool operator<(const Coordinate& coordinate) const
    {
      return r < coordinate.r;
    }

    bool operator>(const Coordinate& coordinate) const
    {
      return r > coordinate.r;
    }

    bool operator==(const Coordinate& coordinate) const
    {
      return q == coordinate.q && r == coordinate.r;
    }

    floatt getX() const
    {
      return r * cos (q);
    }

    floatt getY() const
    {
      return r * sin (q);
    }

    size_t size() const
    {
      return 2;
    }

    floatt at (size_t idx) const
    {
      switch (idx)
      {
        case 0:
          return getX();
        case 1:
          return getY();
      };
      return getY();
    }

    std::string getFormatString (size_t idx) const
    {
      if (label < 0)
      {
        return "r*";
      }
      return "b*";
    }

    int getGeneralLabel () const
    {
      if (label < 0)
      {
        return -1;
      }

      return 1;
    }

    floatt getPreciseLabel () const
    {
      return label;
    }

    void setLabel (floatt label)
    {
      this->label = label;
    }
};

using Coordinates = std::vector<Coordinate>;

template<typename CopyHostMatrixToKernelMatrix, typename GetMatrixInfo>
void runPointsClassification (uintt seed, oap::generic::SingleMatrixProcedures* singleApi, oap::generic::MultiMatricesProcedures* multiApi, oap::NetworkGenericApi* nga,
     CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix, GetMatrixInfo&& getMatrixInfo);

template<typename CopyHostMatrixToKernelMatrix, typename GetMatrixInfo>
void runPointsClassification_multiMatrices (uintt seed, oap::generic::SingleMatrixProcedures* singleApi, oap::generic::MultiMatricesProcedures* multiApi, oap::NetworkGenericApi* nga,
     CopyHostMatrixToKernelMatrix&& copyHostMatrixToKernelMatrix, GetMatrixInfo&& getMatrixInfo);
}

#include "PointsClassification_Test_impl.hpp"
#include "PointsClassification_MultiMatrices_Test_impl.hpp"

#endif
