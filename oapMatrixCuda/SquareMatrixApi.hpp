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

#include "Matrix.hpp"
#include "MatrixInfo.hpp"

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

  math::ComplexMatrix* m_matrix;
  math::ComplexMatrix* m_matrixT;
  math::ComplexMatrix* m_rowVector;
  math::ComplexMatrix* m_subMatrix;
  void destroyMatrices ();

  math::ComplexMatrix* getMatrix ();
  math::ComplexMatrix* getMatrixT ();
  math::ComplexMatrix* getRowVector (uintt index);
  math::ComplexMatrix* getSubMatrix (uintt rindex, uintt rlength);

  void destroyMatrix(math::ComplexMatrix** matrix);
  math::ComplexMatrix* resetMatrix (math::ComplexMatrix* matrix, const math::MatrixInfo& minfo);

  public:
    SquareMatrixApi (CuProceduresApi& api, RecMatrixApi& orig);
    ~SquareMatrixApi ();

    SquareMatrixApi (const SquareMatrixApi& sm) = delete;
    SquareMatrixApi (SquareMatrixApi&& sm) = delete;
    SquareMatrixApi& operator= (const SquareMatrixApi& sm) = delete;
    SquareMatrixApi& operator= (SquareMatrixApi&& sm) = delete;

    math::MatrixInfo getMatrixInfo () const;

    math::ComplexMatrix* createDeviceMatrix ();

    math::ComplexMatrix* getDeviceMatrix (math::ComplexMatrix* dmatrix);
    math::ComplexMatrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::ComplexMatrix* dmatrix);
};

}

#endif
