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
  using AllocationList = oap::AllocationList<const math::Matrix*, math::MatrixInfo>;

  template<typename ExtraUserData>
  using AllocationListEx = oap::AllocationList<const math::Matrix*, std::pair<math::MatrixInfo, ExtraUserData>>;
}

class MatricesList : public AllocationList
{
  public:
    MatricesList (const std::string& id);
    virtual ~MatricesList ();

    virtual std::string toString (const math::MatrixInfo&) const override;
};

template<typename ExtraUserData>
class MatricesListExt : public AllocationListEx<ExtraUserData>
{
  public:
    MatricesListExt (const std::string& id);
    virtual ~MatricesListExt ();

    virtual std::string toString (const std::pair<math::MatrixInfo, ExtraUserData>&) const override;
};

template<typename ExtraUserData>
MatricesListExt<ExtraUserData>::MatricesListExt (const std::string& id) : AllocationListEx<ExtraUserData> (id)
{}

template<typename ExtraUserData>
MatricesListExt<ExtraUserData>::~MatricesListExt ()
{}

template<typename ExtraUserData>
std::string MatricesListExt<ExtraUserData>::toString (const std::pair<math::MatrixInfo, ExtraUserData>& euData) const
{
  return std::to_string (euData.first);
}

#endif
