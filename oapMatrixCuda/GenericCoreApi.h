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

#include "oapTypeTraits.h"
#include "Buffer.h"

#include "Math.h"
#include "Matrix.h"
#include "MatrixInfo.h"
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

  template<typename GetMatrixInfo, typename TransferValueToHost>
  class MatrixMemoryApi
  {
    public:
      MatrixMemoryApi (GetMatrixInfo&& _getMatrixInfo, TransferValueToHost&& _transferValueToHost) :
        getMatrixInfo (std::forward<GetMatrixInfo&&> (_getMatrixInfo)), transferValueToHost (_transferValueToHost)
      {}

      GetMatrixInfo&& getMatrixInfo;
      TransferValueToHost&& transferValueToHost;
  };

  template<typename GetMatrixInfo, typename GetValueIdx, typename SetValueIdx>
  class MatrixApi
  {
    private:
      math::MatrixInfo m_minfo;
      math::Matrix* m_matrix = nullptr;

      inline void updateMatrixInfo (math::Matrix* matrix)
      {
        if (m_matrix != matrix)
        {
          m_minfo = getMatrixInfo (matrix);
          m_matrix = matrix;
        }
      }

    public:

      funcstore<GetMatrixInfo> getMatrixInfo;

      funcstore<GetValueIdx> getReValueIdx;
      funcstore<SetValueIdx> setReValueIdx;

      funcstore<GetValueIdx> getImValueIdx;
      funcstore<SetValueIdx> setImValueIdx;

      MatrixApi (GetMatrixInfo&& _getMatrixInfo, GetValueIdx&& _getReValueIdx, SetValueIdx&& _setReValueIdx, GetValueIdx&& _getImValueIdx, SetValueIdx&& _setImValueIdx) :
        getMatrixInfo (std::forward<GetMatrixInfo&&>(_getMatrixInfo)),
        getReValueIdx (std::forward<GetValueIdx&&>(_getReValueIdx)),
        setReValueIdx (std::forward<SetValueIdx&&>(_setReValueIdx)),
        getImValueIdx (std::forward<GetValueIdx&&>(_getImValueIdx)),
        setImValueIdx (std::forward<SetValueIdx&&>(_setImValueIdx))
      {}

      inline floatt getReValue (math::Matrix* matrix, uintt column, uintt row)
      {
        updateMatrixInfo (matrix);
        debugAssert (m_minfo.isRe);
        return getReValueIdx (matrix, column + row * m_minfo.columns ());
      }

      inline floatt getImValue (math::Matrix* matrix, uintt column, uintt row)
      {
        updateMatrixInfo (matrix);
        debugAssert (m_minfo.isIm);
        return getImValueIdx (matrix, column + row * m_minfo.columns ());
      }

      inline void setReValue (math::Matrix* matrix, uintt column, uintt row, floatt value)
      {
        updateMatrixInfo (matrix);
        debugAssert (m_minfo.isRe);
        setReValueIdx (matrix, column + row * m_minfo.columns (), value);
      }

      inline void setImValue (math::Matrix* matrix, uintt column, uintt row, floatt value)
      {
        updateMatrixInfo (matrix);
        debugAssert (m_minfo.isIm);
        setImValueIdx (matrix, column + row * m_minfo.columns (), value);
      }
  };
}
}

#endif
