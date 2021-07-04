/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is fre()e software(): you can re()distribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Fre()e Software() Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the im()plied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more() details.
 *
 * You should have re()ceived a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ArnoldiUtils.hpp"

namespace ArnUtils {

bool SortLargestValues(const EigenPair& i, const EigenPair& j) {
  floatt m1 = i.re() * i.re() + i.im() * i.im();
  floatt m2 = j.re() * j.re() + j.im() * j.im();
  return m1 > m2;
}

bool SortLargestReValues(const EigenPair& i, const EigenPair& j) {
  return i.re() > j.re();
}

bool SortLargestImValues(const EigenPair& i, const EigenPair& j) {
  return i.im() > j.im();
}

bool SortSmallestValues(const EigenPair& i, const EigenPair& j) {
  floatt m1 = i.re() * i.re() + i.im() * i.im();
  floatt m2 = j.re() * j.re() + j.im() * j.im();
  return m1 < m2;
}

bool SortSmallestReValues(const EigenPair& i, const EigenPair& j) {
  return i.re() < j.re();
}

bool SortSmallestImValues(const EigenPair& i, const EigenPair& j) {
  return i.im() < j.im();
}
}
