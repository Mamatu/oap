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

#ifndef OAP_MATRICES_LIST_H
#define OAP_MATRICES_LIST_H

#include <map>

#include "Matrix.h"
#include "MatrixInfo.h"
#include "oapAllocationList.h"

namespace
{
  using AllocationList = oap::AllocationList<const math::Matrix*, math::MatrixInfo, std::function<std::string(const math::MatrixInfo&)>>;


  template<typename ExtraUserData>
  using UserDataPair = std::pair<math::MatrixInfo, ExtraUserData>;

  template<typename ExtraUserData>
  using AllocationListEx = oap::AllocationList<const math::Matrix*, UserDataPair<ExtraUserData>, std::function<std::string(const UserDataPair<ExtraUserData>&)>>;
}

class MatricesList : public AllocationList
{
  public:
    MatricesList (const std::string& id);
    virtual ~MatricesList ();
};

template<typename ExtraUserData>
class MatricesListExt : public AllocationListEx<ExtraUserData>
{
  public:
    MatricesListExt (const std::string& id);
    virtual ~MatricesListExt ();
};

template<typename ExtraUserData>
MatricesListExt<ExtraUserData>::MatricesListExt (const std::string& id) : AllocationListEx<ExtraUserData> (id, [](const UserDataPair<ExtraUserData>& udp) { return std::to_string (udp.first);})
{}

template<typename ExtraUserData>
MatricesListExt<ExtraUserData>::~MatricesListExt ()
{}
#endif
