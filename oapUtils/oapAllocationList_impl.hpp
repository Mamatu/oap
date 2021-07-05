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

#ifndef OAP_ALLOCATION_LIST__IMPL_H
#define OAP_ALLOCATION_LIST__IMPL_H

#include "oapAllocationList.hpp"
#include "Logger.hpp"

namespace oap
{

template<typename T, typename UserData, typename ToString>
AllocationList<T, UserData, ToString>::AllocationList (const std::string& id, ToString&& _to_string) : m_id(id), to_string(std::forward<ToString>(_to_string))
{}

template<typename T, typename UserData, typename ToString>
AllocationList<T, UserData, ToString>::~AllocationList ()
{
  checkOnDelete ();
}

template<typename T, typename UserData, typename ToString>
const typename AllocationList<T, UserData, ToString>::Map& AllocationList<T, UserData, ToString>::getAllocated() const
{
  return m_existMap;
}

template<typename T, typename UserData, typename ToString>
void AllocationList<T, UserData, ToString>::add (const T object, const UserData& userData)
{
  m_existMap[object] = userData;

  typename Map::iterator it = m_deletedMap.find (object);
  if (it != m_deletedMap.end ())
  {
    m_deletedMap.erase (it);
  }

  logTrace ("Registered in %s scope: object = %p %s", m_id.c_str(), object, to_string (userData).c_str());
}

template<typename T, typename UserData, typename ToString>
UserData AllocationList<T, UserData, ToString>::getUserData (const T object) const
{
  const auto& map = getAllocated();
  auto it = map.find (object);

  debugAssertMsg (it != map.end(), "Matrix %p does not exist or was not allocated in proper way.", object);

  return it->second;
}

template<typename T, typename UserData, typename ToString>
bool AllocationList<T, UserData, ToString>::contains (const T object) const
{
  const auto& map = getAllocated();
  auto it = map.find (object);

	return (it != map.end());
}

template<typename T, typename UserData, typename ToString>
UserData AllocationList<T, UserData, ToString>::remove (const T object)
{
  UserData userData;

  typename Map::iterator it = m_existMap.find(object);
  if (m_existMap.end() != it)
  {
    m_deletedMap[object] = it->second;
    userData = it->second;

    m_existMap.erase(it);
    logTrace ("Unregistered in %s scope: object = %p %s", m_id.c_str(), object, to_string (userData).c_str());
  }
  else
  {
    typename Map::iterator it = m_deletedMap.find(object);
    if (it != m_deletedMap.end ())
    {
      debugError ("Double deallocation in %s: object = %p %s", m_id.c_str(), object, to_string (it->second).c_str());
      debugAssert ("Double deallocation" != nullptr);
    }
    else
    {
      debugError ("Not found in %s: object = %p", m_id.c_str(), object);
      debugAssert ("Object not found" != nullptr);
    }
  }
  return userData;
}

template<typename T, typename UserData, typename ToString>
void AllocationList<T, UserData, ToString>::checkOnDelete()
{
  if (m_existMap.size() > 0)
  {
    debugError ("Memleak: not deallocated matrices");
    for (typename Map::iterator it = m_existMap.begin(); it != m_existMap.end(); ++it)
    {
      debug("Memleak in %s: object = %p %s not deallocated", m_id.c_str(), it->first, to_string (it->second).c_str());
    }
#ifndef OAP_DISABLE_ABORT_MEMLEAK
    debugAssert (false);
#endif
  }
}
}

#endif
