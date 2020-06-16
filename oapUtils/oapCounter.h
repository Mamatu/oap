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

#ifndef OAP_COUNTER_H
#define OAP_COUNTER_H

#include <unordered_map>
#include <functional>
#include "Math.h"
#include "oapAssertion.h"

namespace oap
{
template<typename T1>
using Counter_ContainerType = std::unordered_map<T1, size_t>;

template<typename T, typename CheckOnDelete, T invalid>
class Counter
{
public:
  using CountsType = Counter_ContainerType<T>;

  Counter (CheckOnDelete&& checkOnDelete) : m_checkOnDelete (std::forward<CheckOnDelete>(checkOnDelete))
  {}

  virtual ~Counter ()
  {
    m_checkOnDelete (m_counts);
  }

  uintt increase (T t)
  {
    uintt count = 0;
    if (t == invalid)
    {
      return 0;
    }
    auto it = m_counts.find (t);
    if (it == m_counts.end())
    {
      m_counts[t] = 0;
      it = m_counts.find (t);
    }
    it->second = it->second + 1;
    return it->second;
  }

  uintt decrease (T t)
  {
    uintt count = 0;
    if (t == invalid)
    {
      return 0;
    }
    auto it = m_counts.find (t);
    oapAssert (it != m_counts.end());
    it->second = it->second - 1;
    if (it->second == 0)
    {
      m_counts.erase (it);
      return 0;
    }
    return it->second;
  }

private:
  CountsType m_counts;
  CheckOnDelete m_checkOnDelete;
};
}

#endif
