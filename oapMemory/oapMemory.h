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

#ifndef OAP_MEMORY_MANAGEMENT_H
#define OAP_MEMORY_MANAGEMENT_H

#include <unordered_map>
#include "Logger.h"
#include "oapTypeTraits.h"

namespace oap
{
  template <typename T, typename NewFunc, typename DeleteFunc, T nullvar>
  class MemoryManagement final
  {
    public:
      MemoryManagement (const NewFunc& newFunc, const DeleteFunc& deleteFunc) : m_newFunc (newFunc), m_deleteFunc (deleteFunc)
      {}

      MemoryManagement (NewFunc&& newFunc, DeleteFunc&& deleteFunc) : m_newFunc (std::forward<NewFunc> (newFunc)), m_deleteFunc (std::forward<DeleteFunc> (deleteFunc))
      {}

      ~MemoryManagement ()
      {
        for (auto& kv : m_counts)
        {
          m_deleteFunc (kv.first);
        }

        m_counts.clear ();
      }

      T allocate (size_t length)
      {
        T memory = m_newFunc (length);
        m_counts [memory] = 1;
        return memory;
      }

      bool deallocate (T memory)
      {
        if (memory == nullvar)
        {
          return false;
        }

        auto it = m_counts.find (memory);

        debugAssertMsg (it != m_counts.end(), "Memory %p was not allocated in this class", memory);

        it->second--;

        if (it->second == 0)
        {
          m_deleteFunc (memory);
          m_counts.erase (it);
          return true;
        }

        return false;
      }

      T reuse (T memory)
      {
        if (memory == nullvar)
        {
          return memory;
        }

        auto it = m_counts.find (memory);

        debugAssertMsg (it != m_counts.end(), "Memory %p was not allocated in this class", memory);

        it->second++;
      
        return memory;
      }

    private:
      std::unordered_map<T, size_t> m_counts;
      funcstore<NewFunc> m_newFunc;
      funcstore<DeleteFunc> m_deleteFunc;
  };
}

#endif
