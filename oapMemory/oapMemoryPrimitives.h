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

#ifndef OAP_MEMORY_REGION_H
#define OAP_MEMORY_REGION_H

#include "Math.h"
#include "CuCore.h"

namespace oap
{

struct MemoryLoc
{
  uintt x;
  uintt y;
};

struct MemoryDims
{
  uintt width;
  uintt height;
};

struct Memory
{
  floatt* ptr;
  MemoryDims dims;
};

struct MemoryRegion
{
  MemoryLoc loc;
  MemoryDims dims;
};

struct Memory_3_Args
{
  oap::Memory m_output;
  oap::Memory m_param1;
  oap::Memory m_param2;
};

struct MemoryRegion_3_Args
{
  oap::MemoryRegion m_output;
  oap::MemoryRegion m_param1;
  oap::MemoryRegion m_param2;
};

}

#endif
