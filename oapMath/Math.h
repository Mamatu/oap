/*
 * Copyright 2016 Marcin Matula
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



#ifndef OAP_MATH_TYPES_H
#define OAP_MATH_TYPES_H

//#define NORMAL_TYPES
//#define EXTENDED_TYPES
//#define RICH_TYPES

#ifdef NORMAL_TYPES
#define NORMAL_FLOAT_TYPE
#define NORMAL_INT_TYPES
#endif

#ifdef EXTENDED_TYPES
#define EXTENDED_FLOAT_TYPE
#define EXTENDED_INT_TYPES
#endif

#ifdef RICH_TYPES
#define RICH_FLOAT_TYPE
#define EXTENDED_INT_TYPES
#endif

#ifdef NORMAL_FLOAT_TYPE
typedef float floatt;
#endif

#ifdef EXTENDED_FLOAT_TYPE
typedef double floatt;
#endif

#ifdef RICH_FLOAT_TYPE
typedef long double floatt;
#endif

#ifdef NORMAL_INT_TYPES
typedef int intt;
typedef unsigned int uintt;
#endif

#ifdef EXTENDED_INT_TYPES
typedef long long intt;
typedef unsigned long long uintt;
#endif

#define MATH_VALUE_LIMIT 0.001f

#define MATH_UNDEFINED static_cast<uintt>(-1)

namespace math {
void Memset(floatt* array, floatt value, intt length);
}

struct Complex {
  Complex() {
    re = 0;
    im = 0;
  }

  Complex(floatt re) {
    this->re = re;
    this->im = 0;
  }

  Complex(floatt re, floatt im) {
    this->re = re;
    this->im = im;
  }

  Complex(const Complex& complex) {
    re = complex.re;
    im = complex.im;
  }

  floatt re;
  floatt im;
};

bool operator==(const Complex& c1, const Complex& c2);

#endif /* MATHTYPES_H */
