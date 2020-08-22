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

#ifndef OAP_RECMATRIXAPI_H
#define OAP_RECMATRIXAPI_H

#include "Matrix.h"
#include "oapHostMatrixPtr.h"

namespace oap
{

class RecMatrixApi final
{
    const math::Matrix* m_recHostMatrix;
    const bool m_deallocate;

    oap::HostMatrixPtr m_recSubHostMatrix;

  public:
    RecMatrixApi (const math::Matrix* recHostMatrix, const bool deallocate);
    ~RecMatrixApi ();

    RecMatrixApi (const RecMatrixApi& sm) = delete;
    RecMatrixApi (RecMatrixApi&& sm) = delete;
    RecMatrixApi& operator= (const RecMatrixApi& sm) = delete;
    RecMatrixApi& operator= (RecMatrixApi&& sm) = delete;

    math::Matrix* createDeviceMatrix ();

    math::Matrix* getDeviceMatrix (math::Matrix* dmatrix);

    math::Matrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::Matrix* dmatrix);

    math::MatrixInfo getMatrixInfo () const;

    const math::Matrix* getHostMatrix () const;

    math::Matrix* getHostSubMatrix (uintt cindex, uintt rindex, uintt clength, uintt rlength);
};

}

#endif

