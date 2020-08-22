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

#ifndef OAP_MEMORY_COUNTER_H
#define OAP_MEMORY_COUNTER_H

#include "oapCounter.h"
#include "oapMemoryPrimitives.h"

namespace oap
{

namespace
{
  auto g_checkOnDelete = [](const Counter_ContainerType<floatt*>& counts)
  {
    if (!counts.empty())
    {
      for (auto it = counts.begin(); it != counts.end(); ++it)
      {
        logInfo ("%p %lu", it->first, it->second);
      }
#ifndef OAP_DISABLE_ABORT_MEMLEAK
      oapAssert (!counts.empty());
#endif
    }
  };

  using oapCounter = oap::Counter<floatt*, decltype(g_checkOnDelete), nullptr>;
}

class MemoryCounter : public oapCounter
{
public:
  MemoryCounter () : oapCounter (std::move (g_checkOnDelete))
  {}

  ~MemoryCounter ()
  {}

  uintt increase (floatt* ptr)
  {
    return oapCounter::increase (ptr);
  }

  uintt decrease (floatt* ptr)
  {
    return oapCounter::decrease (ptr);
  }

  uintt increase (const oap::Memory& memory)
  {
    return increase (memory.ptr);
  }

  uintt decrease (const oap::Memory& memory)
  {
    return decrease (memory.ptr);
  }

private:
  std::unordered_map<floatt*, size_t> m_counts;
};
}

#endif
