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

#include "RecToSquareApi.h"

#include "oapCudaMatrixUtils.h"
#include "oapDeviceComplexMatrixUPtr.h"

#include "CuProceduresApi.h"

namespace oap
{

RecToSquareApi::RecToSquareApi (CuProceduresApi& api, const math::ComplexMatrix* recHostMatrix, bool deallocate) :
m_api(nullptr), m_rec (recHostMatrix, deallocate), m_sq (api, m_rec)
{}

RecToSquareApi::RecToSquareApi (const math::ComplexMatrix* recHostMatrix, bool deallocate) :
m_api(new CuProceduresApi()), m_rec (recHostMatrix, deallocate), m_sq (*m_api, m_rec)
{}

RecToSquareApi::~RecToSquareApi()
{
  delete m_api;
}

math::MatrixInfo RecToSquareApi::getMatrixInfo() const
{
  debugFunc ();

  math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  if (minfo.m_matrixDim.columns != minfo.m_matrixDim.rows)
  {
    minfo.m_matrixDim.columns = minfo.m_matrixDim.rows;
  }

  return minfo;
}

math::ComplexMatrix* RecToSquareApi::createDeviceMatrix()
{
  debugFunc ();

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_rec.createDeviceMatrix ();
  }

  return m_sq.createDeviceMatrix ();
}

math::ComplexMatrix* RecToSquareApi::createDeviceSubMatrix(uintt rindex, uintt rlength)
{
  debugFunc ();

  math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  checkIdx (rindex, minfo);
  checkIfZero (rlength);

  minfo.m_matrixDim = {minfo.m_matrixDim.rows, rlength};

  math::ComplexMatrix* doutput = oap::cuda::NewDeviceMatrix (minfo);

  return getDeviceSubMatrix (rindex, rlength, doutput);
}

math::ComplexMatrix* RecToSquareApi::getDeviceSubMatrix (uintt rindex, uintt rlength, math::ComplexMatrix* dmatrix)
{
  debugFunc ();

  if (dmatrix == nullptr)
  {
    return createDeviceSubMatrix (rindex, rlength); 
  }

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  checkIdx (rindex, minfo, dmatrix);
  checkIfZero (rlength, dmatrix);

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_rec.getDeviceSubMatrix (rindex, rlength, dmatrix);
  }

  return m_sq.getDeviceSubMatrix (rindex, rlength, dmatrix);
}

math::ComplexMatrix* RecToSquareApi::createDeviceRowVector (uintt index)
{
  debugFunc ();

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  checkIdx (index, minfo);

  math::ComplexMatrix* doutput = oap::cuda::NewDeviceReMatrix (minfo.m_matrixDim.rows, 1);

  return getDeviceRowVector (index, doutput);
}

math::ComplexMatrix* RecToSquareApi::getDeviceRowVector (uintt index, math::ComplexMatrix* dmatrix)
{
  debugFunc ();

  if (dmatrix == nullptr)
  {
    return createDeviceRowVector (index); 
  }

  const math::MatrixInfo minfo = m_rec.getMatrixInfo ();

  checkIdx (index, minfo, dmatrix);

  if (minfo.m_matrixDim.columns == minfo.m_matrixDim.rows)
  {
    return m_rec.getDeviceSubMatrix (index, 1, dmatrix);
  }

  return m_sq.getDeviceSubMatrix (index, 1, dmatrix);
}

void RecToSquareApi::checkIdx (uintt row, const math::MatrixInfo& minfo, math::ComplexMatrix* matrix) const
{
  if (row >= minfo.rows ())
  {
    oap::cuda::DeleteDeviceMatrix (matrix);

    std::stringstream stream;
    stream << "row out of scope. row: " << row << ", rows: " << minfo.rows ();
    throw std::runtime_error (stream.str());
  }
}

void RecToSquareApi::checkIfZero (uintt length, math::ComplexMatrix* matrix) const
{
  if (length == 0)
  {
    oap::cuda::DeleteDeviceMatrix (matrix);

    std::stringstream stream;
    stream << "length is zero";
    throw std::runtime_error (stream.str());
  }
}

}
