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

#ifndef OAP_SQUAREMATRIXAPI_H
#define OAP_SQUAREMATRIXAPI_H

#include "Matrix.h"
#include "MatrixInfo.h"

namespace oap
{

class CuProceduresApi;
class RecMatrixApi;

class SquareMatrixApi final
{
  CuProceduresApi& m_api;
  RecMatrixApi& m_orig;
  math::MatrixInfo     m_matrixInfo;
  math::MatrixInfo     m_matrixTInfo;
  math::MatrixInfo     m_rowVectorInfo;
  math::MatrixInfo     m_subMatrixInfo;

  math::Matrix* m_matrix;
  math::Matrix* m_matrixT;
  math::Matrix* m_rowVector;
  math::Matrix* m_subMatrix;
  void destroyMatrices ();

  math::Matrix* getMatrix ();
  math::Matrix* getMatrixT ();
  math::Matrix* getRowVector (uintt index);
  math::Matrix* getSubMatrix (uintt rindex, uintt rlength);

  void destroyMatrix(math::Matrix** matrix);
  math::Matrix* resetMatrix (math::Matrix* matrix, const math::MatrixInfo& minfo);

  public:
    SquareMatrixApi (CuProceduresApi& api, RecMatrixApi& orig);
    ~SquareMatrixApi ();

    SquareMatrixApi (const SquareMatrixApi& sm) = delete;
    SquareMatrixApi (SquareMatrixApi&& sm) = delete;
    SquareMatrixApi& operator= (const SquareMatrixApi& sm) = delete;
    SquareMatrixApi& operator= (SquareMatrixApi&& sm) = delete;

    math::MatrixInfo getMatrixInfo () const;

    math::Matrix* createDeviceMatrix ();

    math::Matrix* getDeviceMatrix (math::Matrix* dmatrix);
    math::Matrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::Matrix* dmatrix);
};

}

#endif
