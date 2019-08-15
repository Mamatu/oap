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

#ifndef OAP_GENERIC_CORE_API_H
#define OAP_GENERIC_CORE_API_H

#include <map>
#include <sstream>

#include "Buffer.h"

#include "Math.h"
#include "Matrix.h"
#include "MatrixEx.h"
#include "IKernelExecutor.h"

namespace oap
{
namespace generic
{

  template<typename GetColumns, typename GetRows>
  class BasicMatrixDimApi
  {
    public:
      BasicMatrixDimApi (GetColumns&& _getColumns, GetRows&& _getRows) : getColumns (std::move (_getColumns)), getRows (std::move (_getRows))
      {}

      GetColumns&& getColumns;
      GetRows&& getRows;
  };

  template<typename GetAddress>
  class BasicAddressApi
  {
    public:
      GetAddress&& getReAddress;
      GetAddress&& getImAddress;
  };

  template<typename GetMatrixInfo>
  class BasicMatrixApi
  {
    public:
      BasicMatrixApi (GetMatrixInfo&& _getMatrixInfo) : getMatrixInfo (std::move (_getMatrixInfo)) 
      {}

      GetMatrixInfo&& getMatrixInfo;
  };

  template<typename GetMatrixInfo, typename GetValue>
  class MatrixApi
  {
    public:
      MatrixApi (GetMatrixInfo&& _getMatrixInfo, GetValue&& _getValue) :
        getMatrixInfo (std::forward<GetMatrixInfo&&> (_getMatrixInfo)), getValue (std::forward<GetValue&&> (_getValue))
      {}

      GetMatrixInfo&& getMatrixInfo;
      GetValue&& getValue;
  };
}
}

#endif
