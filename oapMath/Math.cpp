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



#include <string.h>
#include "Math.h"

namespace math {

void Memset(floatt* array, floatt value, intt length) {
  for (uintt fa = 0; fa < length; ++fa) {
    array[fa] = value;
  }
}
}

bool operator==(const Complex& c1, const Complex& c2) {
  floatt limit = 0.2;
  bool isRe = c2.re - limit <= c1.re && c1.re <= c2.re + limit;
  bool isIm = c2.im - limit <= c1.im && c1.im <= c2.im + limit;
  return isRe && isIm;
}
