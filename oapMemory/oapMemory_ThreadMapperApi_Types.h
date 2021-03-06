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

#ifndef OAP_MEMORY__THREAD_MAPPER_API__TYPES_H
#define OAP_MEMORY__THREAD_MAPPER_API__TYPES_H

#include "Math.h"

// Number of indecies which is stored in buffer for one element
#define AIA_INDECIES_COUNT 2
#define MP_INDECIES_COUNT 3

namespace oap
{
namespace threads
{

struct UserData
{
  uintt* mapperBuffer;
  uintt* dataBuffer;
  uintt argsCount;
};

}
}

#endif
