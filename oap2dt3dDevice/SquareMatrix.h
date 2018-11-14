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
#include "oapHostMatrixPtr.h"

namespace oap
{

class SquareMatrix
{
  public:
    SquareMatrix (DeviceDataLoader* ddl);
    explicit SquareMatrix (const math::Matrix* recMatrix, bool deallocate);
    virtual ~SquareMatrix();

    SquareMatrix (const SquareMatrix& sm) = delete;
    SquareMatrix (SquareMatrix&& sm) = delete;
    SquareMatrix& operator= (const SquareMatrix& sm) = delete;
    SquareMatrix& operator= (SquareMatrix&& sm) = delete;

    math::MatrixInfo getMatrixInfo() const;

    math::Matrix* createDeviceMatrix();

    math::Matrix* createDeviceRowVector(uintt index);
    math::Matrix* createDeviceSubMatrix (uintt rindex, uintt rlength);

    math::Matrix* getDeviceRowVector(uintt index, math::Matrix* dmatrix);
    math::Matrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::Matrix* dmatrix);

  private:
    class Orig
    {
        const math::Matrix* m_recHostMatrix;
        const bool m_deallocate;
        oap::HostMatrixPtr m_recSubHostMatrix;

      public:
        Orig (const math::Matrix* recHostMatrix, const bool deallocate);

        ~Orig ();

        math::Matrix* createDeviceMatrix ();

        math::Matrix* getDeviceMatrix (math::Matrix* dmatrix);

        math::Matrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::Matrix* dmatrix);

        math::MatrixInfo getMatrixInfo () const;

        const math::Matrix* getHostMatrix () const;

        math::Matrix* getHostSubMatrix (uintt cindex, uintt rindex, uintt clength, uintt rlength);
    };

    class SqMatrix
    {
      Orig& m_orig;
      math::MatrixInfo     m_matrixInfo;
      math::MatrixInfo     m_matrixTInfo;
      math::MatrixInfo     m_rowVectorInfo;
      math::MatrixInfo     m_subMatrixInfo;

      math::Matrix* m_matrix;
      math::Matrix* m_matrixT;
      math::Matrix* m_rowVector;
      math::Matrix* m_subMatrix;
      void destroyMatrices ();
      CuProceduresApi m_api;

      math::Matrix* getMatrix ();
      math::Matrix* getMatrixT ();
      math::Matrix* getRowVector (uintt index);
      math::Matrix* getSubMatrix (uintt rindex, uintt rlength);

      public:

        SqMatrix (Orig& orig) : m_orig(orig), m_matrix (nullptr), m_matrixT (nullptr), m_rowVector (nullptr), m_subMatrix (nullptr)
        {}

        ~SqMatrix ()
        {
          destroyMatrices ();
        }

        math::MatrixInfo getMatrixInfo () const
        {
          auto minfo = m_orig.getMatrixInfo ();
          minfo.m_matrixDim.columns = minfo.m_matrixDim.rows;
          return minfo;
        }

        math::Matrix* createDeviceMatrix ()
        {
          auto minfo = getMatrixInfo ();
          math::Matrix* matrix = oap::cuda::NewDeviceMatrix (minfo);
          return getDeviceMatrix (matrix);
        }

        math::Matrix* getDeviceMatrix (math::Matrix* dmatrix)
        {
          debugFunc ();
          auto minfo = m_orig.getMatrixInfo ();

          math::Matrix* matrix = getMatrix ();
          math::Matrix* matrixT = getMatrixT ();

          math::Matrix* output = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, minfo.m_matrixDim.rows);

          m_api.dotProduct (output, matrix, matrixT);

          return output;
        }

        math::Matrix* getDeviceSubMatrix (uintt rindex, uintt rlength, math::Matrix* dmatrix)
        {
          debugFunc ();
          auto minfo = m_orig.getMatrixInfo ();

          math::Matrix* matrixT = getMatrixT ();

          math::Matrix* subMatrix = getSubMatrix (rindex, rlength);

          auto subinfo = oap::cuda::GetMatrixInfo (subMatrix);
          auto dinfo = oap::cuda::GetMatrixInfo (dmatrix);

          if (dinfo.m_matrixDim.rows != subinfo.m_matrixDim.rows)
          {
            dinfo.m_matrixDim.rows = subinfo.m_matrixDim.rows;

            oap::cuda::DeleteDeviceMatrix (dmatrix);
            dmatrix = oap::cuda::NewDeviceMatrix (dinfo);
          }

          m_api.dotProduct (dmatrix, subMatrix, matrixT);

          return dmatrix;
        }

        void checkArgs(uintt rindex, uintt rlength, const math::MatrixInfo& minfo)
        {
          if (rindex >= minfo.m_matrixDim.rows)
          {
            destroyMatrices ();
            throw std::runtime_error ("rindex is higher than rows of matrix");
          }

          if (rlength == 0)
          {
            destroyMatrices ();
            throw std::runtime_error ("rlength cannot be zero");
          }
        }
    };

    Orig m_orig;
    SqMatrix m_sq;

    static void destroyMatrix(math::Matrix** matrix);
    static math::Matrix* resetMatrix (math::Matrix* matrix, const math::MatrixInfo& minfo);
};
}

#endif  // SQUAREMATRIX_H
