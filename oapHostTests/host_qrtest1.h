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



#ifndef HOST_QRTEST1_H
#define HOST_QRTEST1_H

namespace host {
namespace qrtest1 {
const char* matrix = "[4,2,2,2,4,2,2,2,4]";

const char* qref =
    "[0.81649658092773, -0.49236596391733, -0.30151134457776, 0.40824829046386, 0.86164043685533, -0.30151134457776, 0.40824829046386, 0.12309149097933, 0.90453403373329]";

const char* rref =
    "[4.8989794855664, 4.0824829046386, 4.0824829046386, 0, 2.7080128015453, 1.2309149097933, 0, 0, 2.4120907566221]";
};
};

#endif  // QRTEST1_H
