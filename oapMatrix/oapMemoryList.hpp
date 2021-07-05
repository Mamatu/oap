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

#ifndef OAP_MEMORY_LIST_H
#define OAP_MEMORY_LIST_H

#include "oapMemory.hpp"
#include "oapMemoryPrimitivesApi.hpp"
#include "oapAllocationList.hpp"

struct MemoryHash
{
  std::size_t operator()(const oap::Memory& memory) const noexcept
  {
    std::size_t index = std::hash<decltype(memory.ptr)>{}(memory.ptr);
    return index;
  }
};

class MemoryList : public oap::AllocationList<floatt*, size_t, std::function<std::string(size_t)>>
{
  public:
    MemoryList (const std::string& id);
    virtual ~MemoryList ();
};

#endif
