/*
 * Copyright 2016, 2017 Marcin Matula
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

#ifndef ARNOLDIUTILS_H
#define ARNOLDIUTILS_H

#include "Math.h"
#include "Matrix.h"

namespace ArnUtils {

bool SortLargestValues(const Complex& i, const Complex& j);

bool SortLargestReValues(const Complex& i, const Complex& j);

bool SortLargestImValues(const Complex& i, const Complex& j);

bool SortSmallestValues(const Complex& i, const Complex& j);

bool SortSmallestReValues(const Complex& i, const Complex& j);

bool SortSmallestImValues(const Complex& i, const Complex& j);

typedef bool (*SortType)(const Complex& i, const Complex& j);

enum CheckType {
  CHECK_INTERNAL,
  CHECK_EXTERNAL,
  CHECK_FIRST_STOP
};

enum Type { UNDEFINED, DEVICE, HOST };
}
#endif  // ARNOLDIUTILS_H
