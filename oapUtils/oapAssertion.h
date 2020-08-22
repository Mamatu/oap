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

// This code is inspired by https://github.com/dmcrodrigues/macro-logger/blob/master/macrologger.h

#ifndef OAP_ASSERTION_H
#define	OAP_ASSERTION_H

#include <assert.h>

#ifdef DEBUG
#define debugAssert(x) assert(x);
#else
#define debugAssert(x)
#endif
#define oapDebugAssert(x) debugAssert(x)

#ifndef OAP_DISABLE_ASSERTION
#define logAssert(x) assert(x);
#else
#define logAssert(x)
#endif

#define logAssertMsg(x, msg, ...) if (!(x)) { fprintf(stderr, msg, ##__VA_ARGS__); abort(); }
#define oapAssert(x) logAssert(x)
#endif
