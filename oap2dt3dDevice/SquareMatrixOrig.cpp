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

#include "SquareMatrix.h"
#include "oapCudaMatrixUtils.h"

#include "oapDeviceMatrixUPtr.h"

namespace oap
{

math::MatrixInfo SquareMatrix::Orig::getMatrixInfo () const
{
  return oap::host::GetMatrixInfo (m_recHostMatrix);
}

SquareMatrix::Orig::Orig (const math::Matrix* recHostMatrix, const bool deallocate) :
  m_recHostMatrix (recHostMatrix), m_deallocate (deallocate)
{}

SquareMatrix::Orig::~Orig ()
{
  if (m_deallocate)
  {
    oap::host::DeleteMatrix (m_recHostMatrix);
  }
}

math::Matrix* SquareMatrix::Orig::createDeviceMatrix ()
{
  auto minfo = getMatrixInfo ();
  math::Matrix* matrix = oap::cuda::NewDeviceMatrix (minfo);
  return getDeviceMatrix (matrix);
}

math::Matrix* SquareMatrix::Orig::getDeviceMatrix (math::Matrix* dmatrix)
{
  oap::cuda::CopyHostMatrixToDeviceMatrix (dmatrix, m_recHostMatrix);
  return dmatrix;
}

math::Matrix* SquareMatrix::Orig::getDeviceSubMatrix (uintt rindex, uintt rlength, math::Matrix* dmatrix)
{
  auto minfo = getMatrixInfo ();
  math::Matrix* subMatrix = getHostSubMatrix (0, rindex, minfo.m_matrixDim.columns, rlength);
  oap::cuda::CopyHostMatrixToDeviceMatrix (dmatrix, subMatrix);
  return dmatrix;
}

const math::Matrix* SquareMatrix::Orig::getHostMatrix () const
{
  return m_recHostMatrix;
}

math::Matrix* SquareMatrix::Orig::getHostSubMatrix (uintt cindex, uintt rindex, uintt clength, uintt rlength)
{
  m_recSubHostMatrix.reset (oap::host::NewSubMatrix (m_recHostMatrix, cindex, rindex, clength, rlength));
  return m_recSubHostMatrix.get ();
}

}
