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



#ifndef SCHUR3
#define SCHUR3

namespace test {
namespace schur3 {
const char* matrix = "[-3.25, 0.901388, 0 <repeats 30 times>,\
        0.901388, -2.32692, 1.23796, 0 <repeats 29 times> \
        0, 1.23796, -1.56883, 1.50634, 0 <repeats 28 times> \
        0 <repeats 2 times>, 1.50634, -1.01633, 1.74628, 0 <repeats 27 times> \
        0 <repeats 3 times>, 1.74628, -0.602154, 1.92572, 0 <repeats 26 times> \
        0 <repeats 4 times>, 1.92572, -0.250931, 2.05745, 0 <repeats 25 times> \
        0 <repeats 5 times>, 2.05745, -0.026333, 2.16272, 0 <repeats 24 times> \
        0 <repeats 6 times>, 2.16272, 0.107838, 2.20957, 0 <repeats 23 times> \
        0 <repeats 7 times>, 2.20957, 0.171008, 2.21548, 0 <repeats 22 times> \
        0 <repeats 8 times>, 2.21548, 0.148167, 2.16397, 0 <repeats 21 times> \
        0 <repeats 9 times>, 2.16397, 0.0908782, 2.04639, 0 <repeats 20 times> \
        0 <repeats 10 times>, 2.04639, 0.0589215, 1.97043, 0 <repeats 19 times> \
        0 <repeats 11 times>, 1.97043, 0.0605254, 2.07195, 0 <repeats 18 times> \
        0 <repeats 12 times>, 2.07195, 0.27312, 2.1432, 0 <repeats 17 times> \
        0 <repeats 13 times>, 2.1432, 0.127581, 2.04269, 0 <repeats 16 times> \
        0 <repeats 14 times>, 2.04269, 0.0939863, 2.04855, 0 <repeats 15 times> \
        0 <repeats 15 times>, 2.04855, 0.0904022, 2.0574, 0 <repeats 14 times> \
        0 <repeats 16 times>, 2.0574, 0.070301, 2.04168, 0 <repeats 13 times> \
        0 <repeats 17 times>, 2.04168, 0.0797113, 2.03044, 0 <repeats 12 times> \
        0 <repeats 18 times>, 2.03044, 0.0368267, 1.98724, 0 <repeats 11 times> \
        0 <repeats 19 times>, 1.98724, -0.0554716, 1.97701, 0 <repeats 10 times> \
        0 <repeats 20 times>, 1.97701, -0.149462, 2.05221, 0 <repeats 9 times> \
        0 <repeats 21 times>, 2.05221, -0.189536, 2.09477, 0 <repeats 8 times> \
        0 <repeats 22 times>, 2.09477, -0.124069, 2.05576, 0 <repeats 7 times> \
        0 <repeats 23 times>, 2.05576, -0.0292181, 1.98898, 0 <repeats 6 times> \
        0 <repeats 24 times>, 1.98898, 0.0580223, 1.97369, 0 <repeats 5 times> \
        0 <repeats 25 times>, 1.97369, 0.052011, 1.94687, 0 <repeats 4 times> \
        0 <repeats 26 times>, 1.94687, 0.0149541, 1.9466, 0 <repeats 3 times> \
        0 <repeats 27 times>, 1.9466, 0.131983, 2.00956, 0 <repeats 2 times> \
        0 <repeats 28 times>, 2.00956, 0.248968, 2.08494, 0 \
        0 <repeats 29 times>, 2.08494, 0.186982, 2.05053 \
        0 <repeats 30 times>, 2.05053, 0.0402749]";
const char* eq_matrix = "[1,-7.1119,-815.8706,0,2,-55.0236,0,0,3]";
}
}

#endif // SCHUR3
