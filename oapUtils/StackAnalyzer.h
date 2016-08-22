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



#ifndef STACKANALYZER_H
#define STACKANALYZER_H

#include <vector>
#include <stdio.h>
#include "Math.h"

namespace stack {
typedef unsigned long long int llu;

void getInstructionsPointers(std::vector<uintt>& rips,
                             std::vector<uintt>& rbps);
}

#endif  // MEMORYANALYZER_H
