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

#ifndef CU_CREATE_PROCEDURES_H
#define CU_CREATE_PROCEDURES_H

#include "CuCore.h"
#include "CuMatrixIndexUtilsCommon.h"
#include "CuCopyProcedures.h"

#include "Matrix.h"
#include "MatrixEx.h"

#include "oapAssertion.h"

#define CuM_InitMatrix(m) m.re.mem = {NULL, {0, 0}}; m.im.mem = {NULL, {0, 0}}; m.re.reg = {{0, 0}, {0, 0}}; m.im.reg = {{0, 0}, {0, 0}};

struct MatrixOffset
{
  math::ComplexMatrix matrix;
  floatt* buffer;
};

__hostdeviceinline__ MatrixOffset CUDA_createGenericMatrixCopy (bool isRe, bool isIm, uintt columns, uintt rows, floatt* buffer, const math::ComplexMatrix* matrix, const MatrixEx& matrixEx)
{
  HOST_INIT();
  THREAD_INDICES_INIT();

  debugAssert (!isRe || isRe == (matrix->re.mem.ptr != NULL));
  debugAssert (!isIm || isIm == (matrix->im.mem.ptr != NULL));
  debugAssert (columns == matrixEx.columns);
  debugAssert (rows == matrixEx.rows);

  math::ComplexMatrix oMatrix;

  oMatrix.dim.columns = columns;
  oMatrix.dim.rows = rows;

  oMatrix.re.mem.ptr = NULL; 
  oMatrix.re.mem.dims.width = 0; 
  oMatrix.re.mem.dims.height = 0; 

  oMatrix.re.reg.loc.x = 0;
  oMatrix.re.reg.loc.y = 0;
  oMatrix.re.reg.dims.width = 0;
  oMatrix.re.reg.dims.height = 0;

  oMatrix.im.mem.ptr = NULL; 
  oMatrix.im.mem.dims.width = 0; 
  oMatrix.im.mem.dims.height = 0; 

  oMatrix.im.reg.loc.x = 0;
  oMatrix.im.reg.loc.y = 0;
  oMatrix.im.reg.dims.width = 0;
  oMatrix.im.reg.dims.height = 0;

  uintt offset = 0;

  if (isRe)
  {
    oMatrix.re.mem.ptr = &buffer[offset];
    oMatrix.re.mem.dims.width = columns;
    oMatrix.re.mem.dims.height = rows;
    offset += rows * columns;
  }

  if (isIm)
  {
    oMatrix.im.mem.ptr = (floatt*)&buffer[offset];
    oMatrix.im.mem.dims.width = columns;
    oMatrix.im.mem.dims.height = rows;
    offset += rows * columns;
  }

  CUDA_copyMatrixEx (&oMatrix, matrix, matrixEx);

  return {oMatrix, &buffer[offset]};
}

__hostdeviceinline__ MatrixOffset CUDA_createReMatrixCopy (floatt* buffer, const math::ComplexMatrix* matrix, const MatrixEx& matrixEx)
{
  return CUDA_createGenericMatrixCopy (true, false, matrixEx.columns, matrixEx.rows, buffer, matrix, matrixEx);
}

__hostdeviceinline__ MatrixOffset CUDA_createImMatrixCopy (floatt* buffer, const math::ComplexMatrix* matrix, const MatrixEx& matrixEx)
{
  return CUDA_createGenericMatrixCopy (false, true, matrixEx.columns, matrixEx.rows, buffer, matrix, matrixEx);
}

__hostdeviceinline__ MatrixOffset CUDA_createRealMatrixCopy (floatt* buffer, const math::ComplexMatrix* matrix, const MatrixEx& matrixEx)
{
  return CUDA_createGenericMatrixCopy (true, true, matrixEx.columns, matrixEx.rows, buffer, matrix, matrixEx);
}

__hostdeviceinline__ MatrixOffset CUDA_createMatrixCopy (floatt* buffer, const math::ComplexMatrix* matrix, const MatrixEx& matrixEx)
{
  const bool isRe = matrix->re.mem.ptr != NULL;
  const bool isIm = matrix->im.mem.ptr != NULL;

  if (isRe && isIm)
  {
    return CUDA_createRealMatrixCopy (buffer, matrix, matrixEx);
  }
  else if (isRe)
  {
    return CUDA_createReMatrixCopy (buffer, matrix, matrixEx);
  }

  return CUDA_createImMatrixCopy (buffer, matrix, matrixEx);
}

#endif
