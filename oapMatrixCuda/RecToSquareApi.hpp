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

#ifndef OAP_RECTOSQUAREAPI_H
#define OAP_RECTOSQUAREAPI_H

#include "Matrix.hpp"

#include "RecMatrixApi.hpp"
#include "SquareMatrixApi.hpp"

namespace oap
{

class CuProceduresApi;

class RecToSquareApi final
{
  public:
    explicit RecToSquareApi (CuProceduresApi& api, const math::ComplexMatrix* recMatrix, bool deallocate);
    explicit RecToSquareApi (const math::ComplexMatrix* recMatrix, bool deallocate);
    virtual ~RecToSquareApi();

    RecToSquareApi (const RecToSquareApi& sm) = delete;
    RecToSquareApi (RecToSquareApi&& sm) = delete;
    RecToSquareApi& operator= (const RecToSquareApi& sm) = delete;
    RecToSquareApi& operator= (RecToSquareApi&& sm) = delete;

    math::MatrixInfo getMatrixInfo() const;

    math::ComplexMatrix* createDeviceMatrix();

    math::ComplexMatrix* createDeviceRowVector(uintt index);
    math::ComplexMatrix* createDeviceSubMatrix (uintt rindex, uintt rlength);

    math::ComplexMatrix* getDeviceRowVector(uintt index, math::ComplexMatrix* dmatrix);
    math::ComplexMatrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::ComplexMatrix* dmatrix);

  private:
    CuProceduresApi* m_api;
    RecMatrixApi m_rec;
    SquareMatrixApi m_sq;

    void checkIdx (uintt row, const math::MatrixInfo& minfo, math::ComplexMatrix* matrix = nullptr) const;
    void checkIfZero (uintt length, math::ComplexMatrix* matrix = nullptr) const;
};

}

#endif  // SQUAREMATRIX_H
