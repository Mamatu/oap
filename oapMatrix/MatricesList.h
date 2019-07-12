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

#include "Logger.h"

class MatricesList
{
  public:
    using MatrixInfos = std::map<const math::Matrix*, math::MatrixInfo>;

    MatricesList (const std::string& id);

    MatricesList (const MatricesList&) = delete;
    MatricesList (MatricesList&&) = delete;
    MatricesList& operator= (const MatricesList&) = delete;
    MatricesList& operator= (MatricesList&&) = delete;

    virtual ~MatricesList ();

    const MatrixInfos& getAllocated() const;

    void add (math::Matrix* matrix, const math::MatrixInfo& minfo);

    math::MatrixInfo getMatrixInfo (const math::Matrix* matrix) const;

		bool contains (const math::Matrix* matrix) const;

    math::MatrixInfo remove (const math::Matrix* matrix);

  private:
    std::string m_id;

    MatrixInfos m_matrixInfos;
    MatrixInfos m_deletedMatrixInfos;

    void checkOnDelete();
};

#endif

