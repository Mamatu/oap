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

#include "oapEigen.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>

#include "Matrix.hpp"
#include "oapHostMatrixPtr.hpp"
#include "oapHostComplexMatrixPtr.hpp"

#include "oapHostMatrixApi.hpp"
#include "oapHostComplexMatrixApi.hpp"

namespace math
{

namespace
{
using EMatrixXd = ::Eigen::Matrix<floatt, ::Eigen::Dynamic, ::Eigen::Dynamic, ::Eigen::RowMajor>;
using EMatrixXcd = ::Eigen::Matrix<std::complex<floatt>, ::Eigen::Dynamic, ::Eigen::Dynamic, ::Eigen::RowMajor>;

using MMatrixXd = ::Eigen::Map<EMatrixXd>;
using MMatrixXcd = ::Eigen::Map<EMatrixXcd>;

template<typename MatrixPtr, typename MatrixT, typename NewMatrix, typename CopyMatrix>
MatrixPtr cutToDim(MatrixT* matrix, uintt columns, uintt rows, NewMatrix&& newMatrix, CopyMatrix&& copyMatrix)
{
  if (matrix->reg.dims.width == columns && matrix->reg.dims.height == rows)
  {
    return MatrixPtr(matrix, false);
  }
  math::Matrix* nMatrix = newMatrix (columns, rows);
  copyMatrix(nMatrix, matrix);
  return MatrixPtr (nMatrix);
}

oap::ConstHostMatrixPtr cutToDim(const Matrix* matrix, uintt columns, uintt rows)
{
  return cutToDim<oap::ConstHostMatrixPtr, const oap::math::Matrix> (matrix, columns, rows, oap::host::NewMatrix, oap::host::CopyHostMatrixToHostMatrix);
}

MMatrixXd toEigen(const math::Matrix* matrix)
{
  const uintt columns = oap::host::GetColumns (matrix);
  const uintt rows = oap::host::GetRows (matrix);

  auto matrixPtr = cutToDim(matrix, columns, rows);

  MMatrixXd mmatrix (matrixPtr->mem.ptr, rows, columns);

  return mmatrix;
}

EMatrixXcd toEigen(const math::ComplexMatrix* matrix)
{
  const uintt columns = oap::chost::GetColumns (matrix);
  const uintt rows = oap::chost::GetRows (matrix);

  auto matrixRePtr = cutToDim(&matrix->re, columns, rows);
  auto matrixImPtr = cutToDim(&matrix->im, columns, rows);

  MMatrixXd re (matrixRePtr->mem.ptr, rows, columns);
  MMatrixXd im (matrixImPtr->mem.ptr, rows, columns);

  EMatrixXcd cMatrix(rows, columns);

  cMatrix.real() = re;
  cMatrix.imag() = im;

  return cMatrix;
}
}

void Eigen::magnitude (floatt& output, const math::Matrix* param)
{
  return norm(output, param);
}

void Eigen::magnitude (std::complex<floatt>& output, const math::ComplexMatrix* param)
{
  return norm(output, param);
}

void Eigen::norm (floatt& output, const math::Matrix* param)
{
  EMatrixXd eparam = toEigen (param);
  auto real = eparam.norm();
  output = real;
}

void Eigen::norm (std::complex<floatt>& output, const math::ComplexMatrix* param)
{
  EMatrixXcd eparam = toEigen (param);
  auto real = eparam.norm();
  output = real;
}

void Eigen::squareNorm (floatt& output, const math::Matrix* param)
{
  EMatrixXd eparam = toEigen (param);
  auto real = eparam.norm();
  output = real;
}

void Eigen::squareNorm (std::complex<floatt>& output, const math::ComplexMatrix* param)
{
  EMatrixXcd eparam = toEigen (param);
  auto real = eparam.norm();
  output = real;
}

void Eigen::add (math::Matrix* output, const math::Matrix* param1, const math::Matrix* param2)
{
  //::Eigen::MatrixXd eparam1 = toEigen (param1);
  //::Eigen::MatrixXd eparam2 = toEigen (param2);
}

void Eigen::add (math::ComplexMatrix* output, const math::ComplexMatrix* param1, const math::ComplexMatrix* param2)
{

}
}
