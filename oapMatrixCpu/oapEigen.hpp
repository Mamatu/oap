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

#ifndef OAP_EIGENIMPL_HPP
#define	OAP_EIGENIMPL_HPP

#include <complex>
#include <map> 
#include <vector> 

#include "Matrix.hpp"

namespace math
{
class Eigen final
{
public:
  Eigen() = default;
  ~Eigen() = default;

  void magnitude (floatt& output, const math::Matrix* param);
  void magnitude (std::complex<floatt>& output, const math::ComplexMatrix* param);
 
  void norm (floatt& output, const math::Matrix* param);
  void norm (std::complex<floatt>& output, const math::ComplexMatrix* param);

  void squareNorm (floatt& output, const math::Matrix* param);
  void squareNorm (std::complex<floatt>& output, const math::ComplexMatrix* param);
  
  void add (math::Matrix* output, const math::Matrix* param1, const math::Matrix* param2);
  void add (math::ComplexMatrix* output, const math::ComplexMatrix* param1, const math::ComplexMatrix* param2);

};

}
#endif
