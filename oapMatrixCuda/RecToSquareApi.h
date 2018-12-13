/*
 * Copyright 2016 - 2018 Marcin Matula
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

#include "Matrix.h"

#include "RecMatrixApi.h"
#include "SquareMatrixApi.h"

namespace oap
{

class CuProceduresApi;

class RecToSquareApi
{
  public:
    explicit RecToSquareApi (CuProceduresApi& api, const math::Matrix* recMatrix, bool deallocate);
    explicit RecToSquareApi (const math::Matrix* recMatrix, bool deallocate);
    virtual ~RecToSquareApi();

    RecToSquareApi (const RecToSquareApi& sm) = delete;
    RecToSquareApi (RecToSquareApi&& sm) = delete;
    RecToSquareApi& operator= (const RecToSquareApi& sm) = delete;
    RecToSquareApi& operator= (RecToSquareApi&& sm) = delete;

    math::MatrixInfo getMatrixInfo() const;

    math::Matrix* createDeviceMatrix();

    math::Matrix* createDeviceRowVector(uintt index);
    math::Matrix* createDeviceSubMatrix (uintt rindex, uintt rlength);

    math::Matrix* getDeviceRowVector(uintt index, math::Matrix* dmatrix);
    math::Matrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::Matrix* dmatrix);

  private:
    CuProceduresApi* m_api;
    RecMatrixApi m_rec;
    SquareMatrixApi m_sq;

    void checkIdx (uintt row, const math::MatrixInfo& minfo, math::Matrix* matrix = nullptr) const;
    void checkIfZero (uintt length, math::Matrix* matrix = nullptr) const;
};

}

#endif  // SQUAREMATRIX_H
