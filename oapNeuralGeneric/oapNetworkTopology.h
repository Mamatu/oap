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

#ifndef OAP_NEURAL_NETWORK_TOPOLOGY_H
#define OAP_NEURAL_NETWORK_TOPOLOGY_H

#include <unordered_map>

#include "oapHandler.h"
#include "oapLayer.h"
#include "oapLayerStructure.h"
#include "oapProcedures.h"
#include "oapNetworkGenericApi.h"
#include "oapHostComplexMatrixPtr.h"
#include "oapHostComplexMatrixUPtr.h"

namespace oap
{

using MHandler = oap::Handler<1>;

class InputTopology
{
  public:
    struct MStruct
    {
      oap::math::MatrixInfo minfo;

      MHandler handler;
      oap::math::MatrixLoc loc;
      oap::math::MatrixDim dim;
    };

    class Data
    {
      private:
        std::vector<MStruct> mhandlers;
        std::vector<std::vector<MHandler>> lhandlers;
    
      public:
        uintt getMStructsCount() const;
        uintt getInputLayersCount() const;

        MStruct getMStructByIdx (uintt idx) const;
        MStruct getMStructByHandler (MHandler handler) const;

        std::vector<MHandler> getInputLayer (uintt idx) const;
        std::vector<MStruct> getInputLayerMatrices (uintt idx) const;
        friend class InputTopology;
    };

    LHandler addVertical(const math::MatrixInfo& minfo, uintt count);
    LHandler addVertical(uintt rows, uintt count);

    LHandler addSharedVertical(const math::MatrixInfo& minfo, uintt count, MHandler mhandler);
    LHandler addSharedVertical(uintt rows, uintt count, MHandler mhandler);

    MHandler addMatrix(const math::MatrixInfo& minfo, LHandler lhandler);
    MHandler addSharedMatrix(const math::MatrixInfo& minfo, LHandler lhandler, MHandler sharingMatrix, const oap::math::MatrixLoc& loc, const oap::math::MatrixDim& dim);

    const Data& getData () const;

  private:
    static oap::math::ComplexMatrix getMatrixRef(const math::MatrixInfo& minfo);

    Data m_data;
};
}

#endif
