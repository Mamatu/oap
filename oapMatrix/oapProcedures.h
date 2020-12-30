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

#ifndef OAP_GENERIC_PROCEDURES_H
#define OAP_GENERIC_PROCEDURES_H

#include <vector>
#include <array>

#include "Matrix.h"

namespace oap
{
namespace generic
{

using Dim32 = std::array<std::array<uintt, 2>, 3>;
using Dim22 = std::array<std::array<uintt, 2>, 2>;
using Dim2 = std::array<uintt, 2>;

class SingleMatrixProcedures
{
  public:
    virtual void sigmoid (math::Matrix* output, math::Matrix* input) = 0;

    virtual void linear (math::Matrix* output, math::Matrix* input) = 0;

    virtual void tanh (math::Matrix* output, math::Matrix* input) = 0;

    virtual void sin (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void relu (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void prelu (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void softplus (math::Matrix* output, math::Matrix* input) = 0;

    virtual void sigmoid (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void linear (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void tanh (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void sin (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void relu (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void prelu (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void softplus (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void sigmoid (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void linear (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void tanh (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void sin (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void relu (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void prelu (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void softplus (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void dsigmoid (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void dlinear (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void dtanh (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void dsin (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void drelu (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void dprelu (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void dsoftplus (math::Matrix* output, math::Matrix* input) = 0;
  
    virtual void dsigmoid (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void dlinear (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void dtanh (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void dsin (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void drelu (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void dprelu (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void dsoftplus (math::Matrix* output, math::Matrix* input, Dim2 dim) = 0;
  
    virtual void dsigmoid (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void dlinear (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void dtanh (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void dsin (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void drelu (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void dprelu (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void dsoftplus (math::Matrix* output, math::Matrix* input, Dim22 dim) = 0;
  
    virtual void dotProduct (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, Dim32 dim) = 0;
  
    virtual void dotProductDimPeriodic (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, Dim32 dim, uintt periodicRows) = 0;
  
    virtual void crossEntropy (math::Matrix* output, math::Matrix* param1, math::Matrix* param2) = 0;
  
    virtual void subtract (math::Matrix* output, math::Matrix* param1, math::Matrix* param2) = 0;
  
    virtual void hadamardProductVec (math::Matrix* output, math::Matrix* param1, math::Matrix* param2) = 0;
  
    virtual void transpose (math::Matrix* output, math::Matrix* param1) = 0;
  
    virtual void tensorProduct (math::Matrix* output, math::Matrix* param1, math::Matrix* param2, Dim32 dim) = 0;
  
    virtual void add (math::Matrix* output, math::Matrix* param1, math::Matrix* param2) = 0;
  
    virtual void multiplyReConstant (math::Matrix* output, math::Matrix* param1, floatt re) = 0;
  
    virtual void sum (floatt& output, const math::Matrix* param) = 0;
  
    virtual void setZeroMatrix (math::Matrix* param) = 0;
};

class MultiMatricesProcedures
{
  public:
    using Matrices = std::vector<math::Matrix*>;
  
    virtual void sigmoid (Matrices& output, const Matrices& input) = 0;
  
    virtual void linear (Matrices& output, const Matrices& input) = 0;
  
    virtual void tanh (Matrices& output, const Matrices& input) = 0;
  
    virtual void sin (Matrices& output, const Matrices& input) = 0;
  
    virtual void relu (Matrices& output, const Matrices& input) = 0;
  
    virtual void prelu (Matrices& output, const Matrices& input) = 0;
  
    virtual void softplus (Matrices& output, const Matrices& input) = 0;
  
    virtual void dsigmoid (Matrices& output, const Matrices& input) = 0;
  
    virtual void dlinear (Matrices& output, const Matrices& input) = 0;
  
    virtual void dtanh (Matrices& output, const Matrices& input) = 0;
  
    virtual void dsin (Matrices& output, const Matrices& input) = 0;
  
    virtual void drelu (Matrices& output, const Matrices& input) = 0;
  
    virtual void dprelu (Matrices& output, const Matrices& input) = 0;
  
    virtual void dsoftplus (Matrices& output, const Matrices& input) = 0;
  
    virtual void dotProduct (Matrices& output, const Matrices& param1, const Matrices& param2) = 0;
  
    virtual void subtract (Matrices& output, const Matrices& param1, const Matrices& param2) = 0;
  
    virtual void hadamardProductVec (Matrices& output, const Matrices& param1, const Matrices& param2) = 0;
  
    virtual void transpose (Matrices& output, const Matrices& param1) = 0;
};

}
}
#endif
