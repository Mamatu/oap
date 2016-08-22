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



#include "NewOperator.h"
#include <vector>
#include <map>
/*
typedef std::pair<std::vector<uintt>, std::vector<uintt> > StackPointers;

class Operators {
 public:
  static std::map<void*, StackPointers> m_pointers;
};

std::map<void*, StackPointers> Operators::m_pointers;

void* operator new(std::size_t count) throw std::bad_alloc {
  void* ptr = ::operator new(count);
  return ptr;
}

void* operator new[](std::size_t count) throw std::bad_alloc {
  void* ptr = ::operator new[](count);
  return ptr;
}

void operator delete(void* ptr) throw std::bad_alloc { ::operator delete(ptr); }

void operator delete[](void* ptr) throw std::bad_alloc {
  ::operator delete[](ptr);
}
*/
