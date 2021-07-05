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

#include "Matrix.hpp"
#include "oapHostComplexMatrixPtr.hpp"

namespace oap
{

class RecMatrixApi final
{
    const math::ComplexMatrix* m_recHostMatrix;
    const bool m_deallocate;

    oap::HostComplexMatrixPtr m_recSubHostMatrix;

  public:
    RecMatrixApi (const math::ComplexMatrix* recHostMatrix, const bool deallocate);
    ~RecMatrixApi ();

    RecMatrixApi (const RecMatrixApi& sm) = delete;
    RecMatrixApi (RecMatrixApi&& sm) = delete;
    RecMatrixApi& operator= (const RecMatrixApi& sm) = delete;
    RecMatrixApi& operator= (RecMatrixApi&& sm) = delete;

    math::ComplexMatrix* createDeviceMatrix ();

    math::ComplexMatrix* getDeviceMatrix (math::ComplexMatrix* dmatrix);

    math::ComplexMatrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::ComplexMatrix* dmatrix);

    math::MatrixInfo getMatrixInfo () const;

    const math::ComplexMatrix* getHostMatrix () const;

    math::ComplexMatrix* getHostSubMatrix (uintt cindex, uintt rindex, uintt clength, uintt rlength);
};

}

#endif

