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

#include "oapProcedures.h"

namespace oap
{
namespace generic
{
/*
void SingleMatrixProcedures::sigmoid (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::linear (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::tanh (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::sin (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::relu (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::prelu (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::softplus (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::sigmoid (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::linear (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::tanh (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::sin (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::relu (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::prelu (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::softplus (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsigmoid (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dlinear (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dtanh (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsin (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::drelu (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dprelu (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsoftplus (math::Matrix* output, math::Matrix* input)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsigmoid (math::Matrix* output, math::Matrix* input, Dim2 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dlinear (math::Matrix* output, math::Matrix* input, Dim2 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dtanh (math::Matrix* output, math::Matrix* input, Dim2 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsin (math::Matrix* output, math::Matrix* input, Dim2 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::drelu (math::Matrix* output, math::Matrix* input, Dim2 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dprelu (math::Matrix* output, math::Matrix* input, Dim2 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsoftplus (math::Matrix* output, math::Matrix* input, Dim2 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsigmoid (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dlinear (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dtanh (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsin (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::drelu (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dprelu (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dsoftplus (math::Matrix* output, math::Matrix* input, Dim22 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dotProduct (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, Dim32 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::dotProductDimPeriodic (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, Dim32 dim, uintt periodicRows)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::crossEntropy (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::subtract (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::hadamardProductVec (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::transpose (math::Matrix* output, math::Matrix* param1)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::tensorProduct (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, Dim32 dim)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::add (math::Matrix* output, math::Matrix* param1, math::Matrix* param2)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::multiplyReConstant (math::Matrix* output, math::Matrix* param1, floatt re)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::sum (floatt& output, math::Matrix* param)
{
  oapAssert ("Not implemented" == nullptr);
}

void SingleMatrixProcedures::setZeroMatrix (math::Matrix* param)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::sigmoid (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::linear (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::tanh (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::sin (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::relu (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::prelu (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::softplus (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::dsigmoid (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::dlinear (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::dtanh (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::dsin (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::drelu (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::dprelu (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::dsoftplus (Matrices& output, const Matrices& input)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::dotProduct (Matrices& output, const Matrices& param1, const Matrices& param2)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::subtract (Matrices& output, const Matrices& param1, const Matrices& param2)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::hadamardProductVec (Matrices& output, const Matrices& param1, const Matrices& param2)
{
  oapAssert ("Not implemented" == nullptr);
}

void MultiMatricesProcedures::transpose (Matrices& output, const Matrices& param1)
{
  oapAssert ("Not implemented" == nullptr);
}
*/
}
}
