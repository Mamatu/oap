/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef OAP_MATH_FUNCTIONS
#define OAP_MATH_FUNCTIONS

#include <cmath>
#include <numeric>
#include <vector>

#include "Math.h"

namespace oap { namespace math {

  inline floatt sigmoid(floatt x)
  {
    return 1.f / (1.f + exp (-x));
  }

  inline floatt dsigmoid(floatt x)
  {
    return sigmoid(x) * (1.f - sigmoid(x));
  }

  inline floatt sum(const std::vector<floatt>& values)
  {
    return std::accumulate(values.begin(), values.end(), static_cast<floatt>(0));
  }

  inline floatt tanh(floatt x)
  {
    return ::tanh (x);
  }

  inline floatt sin(floatt x)
  {
    return ::sin (x);
  }
}
}

#endif
