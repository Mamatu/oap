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

#ifndef OAP_SQUAREMATRIX_H
#define OAP_SQUAREMATRIX_H

#include "DeviceDataLoader.h"
#include "CuProceduresApi.h"

#include "oapDeviceMatrixPtr.h"

namespace oap
{

class SquareMatrix
{
  public:
    SquareMatrix (DeviceDataLoader* ddl);

    virtual ~SquareMatrix();

    math::MatrixInfo getMatrixInfo() const;

    math::Matrix* createDeviceMatrix();

    math::Matrix* createDeviceRowVector(size_t index);
    math::Matrix* createDeviceSubMatrix (size_t rindex, size_t rlength);

    math::Matrix* getDeviceRowVector(size_t index, math::Matrix* dmatrix);
    math::Matrix* getDeviceSubMatrix (size_t rindex, size_t rlength, math::Matrix* dmatrix);

  private:
    DeviceDataLoader* m_ddl;
    CuProceduresApi m_api;

    math::MatrixInfo     m_matrixInfo;
    math::MatrixInfo     m_matrixTInfo;
    math::MatrixInfo     m_rowVectorInfo;
    math::MatrixInfo     m_subMatrixInfo;
    math::Matrix* m_matrix;
    math::Matrix* m_matrixT;
    math::Matrix* m_rowVector;
    math::Matrix* m_subMatrix;

    math::Matrix* constructSquareMatrix (const math::MatrixInfo& minfo);
    math::Matrix* getMatrix (const math::MatrixInfo& minfo);
    math::Matrix* getMatrixT (const math::MatrixInfo& minfo);
    math::Matrix* getRowVector (size_t index, const math::MatrixInfo& minfo);
    math::Matrix* getSubMatrix (size_t rindex, size_t rlength, const math::MatrixInfo& minfo);

    void destroyMatrix(math::Matrix** matrix);
    void destroyMatrices ();

    void checkRIndex(uintt rindex, const math::MatrixInfo& minfo)
    {
      if (rindex >= minfo.m_matrixDim.rows)
      {
        destroyMatrices ();
        throw std::runtime_error ("rindex is higher than rows of matrix");
      }
    }
};
}

#endif  // SQUAREMATRIX_H
