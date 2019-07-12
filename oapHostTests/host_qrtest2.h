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



#ifndef HOST_QRTEST2_H
#define HOST_QRTEST2_H

namespace host {
namespace qrtest2 {
const char* matrix = "[4,2,2,4,2,4,2,4,2,2,4,4,2,2,4,4]";

const char* qref =
    "[0.75593, -0.52414, -0.39223,  0.00000,\
        0.37796,  0.83863, -0.39223,  0.00000,\
        0.37796,  0.10483,  0.58835, -0.70711,\
        0.37796,  0.10483,  0.58835,  0.70711]";

const char* rref =
    "[5.29150, 4.53557, 5.29150, 7.55929,\
        0.00000, 2.72554, 1.46760, 2.09657,\
        0.00000, 0.00000, 3.13786, 1.56893,\
        0.00000, 0.00000, 0.00000, 0.00000]";
};
};

#endif  // QRTEST2
