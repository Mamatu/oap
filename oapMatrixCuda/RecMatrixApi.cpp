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

#include "RecMatrixApi.hpp"

#include "oapCudaMatrixUtils.hpp"
#include "oapDeviceComplexMatrixUPtr.hpp"

#include "CuProceduresApi.hpp"

namespace oap
{

math::MatrixInfo RecMatrixApi::getMatrixInfo () const
{
  return oap::chost::GetMatrixInfo (m_recHostMatrix);
}

RecMatrixApi::RecMatrixApi (const math::ComplexMatrix* recHostMatrix, const bool deallocate) :
  m_recHostMatrix (recHostMatrix), m_deallocate (deallocate)
{}

RecMatrixApi::~RecMatrixApi ()
{
  if (m_deallocate)
  {
    oap::chost::DeleteMatrix (m_recHostMatrix);
  }
}

math::ComplexMatrix* RecMatrixApi::createDeviceMatrix ()
{
  auto minfo = getMatrixInfo ();
  math::ComplexMatrix* matrix = oap::cuda::NewDeviceMatrix (minfo);
  return getDeviceMatrix (matrix);
}

math::ComplexMatrix* RecMatrixApi::getDeviceMatrix (math::ComplexMatrix* dmatrix)
{
  oap::cuda::CopyHostMatrixToDeviceMatrix (dmatrix, m_recHostMatrix);
  return dmatrix;
}

math::ComplexMatrix* RecMatrixApi::getDeviceSubMatrix (uintt rindex, uintt rlength, math::ComplexMatrix* dmatrix)
{
  auto minfo = getMatrixInfo ();
  math::ComplexMatrix* subMatrix = getHostSubMatrix (0, rindex, minfo.m_matrixDim.columns, rlength);
  oap::cuda::CopyHostMatrixToDeviceMatrix (dmatrix, subMatrix);
  return dmatrix;
}

const math::ComplexMatrix* RecMatrixApi::getHostMatrix () const
{
  return m_recHostMatrix;
}

math::ComplexMatrix* RecMatrixApi::getHostSubMatrix (uintt cindex, uintt rindex, uintt clength, uintt rlength)
{
  m_recSubHostMatrix = (oap::chost::NewSubMatrix (m_recHostMatrix, cindex, rindex, clength, rlength));
  return m_recSubHostMatrix.get ();
}

}
