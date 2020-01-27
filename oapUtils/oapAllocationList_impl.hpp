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

#include "oapAllocationList.h"
#include "Logger.h"

namespace oap
{

template<typename T, typename UserData>
AllocationList<T, UserData>::AllocationList (const std::string& id) : m_id(id)
{}

template<typename T, typename UserData>
AllocationList<T, UserData>::~AllocationList ()
{
  checkOnDelete ();
}

template<typename T, typename UserData>
const typename AllocationList<T, UserData>::Map& AllocationList<T, UserData>::getAllocated() const
{
  return m_existMap;
}

template<typename T, typename UserData>
void AllocationList<T, UserData>::add (const T object, const UserData& userData)
{
  m_existMap[object] = userData;

  typename Map::iterator it = m_deletedMap.find (object);
  if (it != m_deletedMap.end ())
  {
    m_deletedMap.erase (it);
  }

  logTrace ("Registered in %s scope: object = %p %s", m_id.c_str(), object, std::to_string (userData).c_str());
}

template<typename T, typename UserData>
UserData AllocationList<T, UserData>::getUserData (const T object) const
{
  const auto& map = getAllocated();
  auto it = map.find (object);

  debugAssertMsg (it != map.end(), "Matrix %p does not exist or was not allocated in proper way.", object);

  return it->second;
}

template<typename T, typename UserData>
bool AllocationList<T, UserData>::contains (const T object) const
{
  const auto& map = getAllocated();
  auto it = map.find (object);

	return (it != map.end());
}

template<typename T, typename UserData>
UserData AllocationList<T, UserData>::remove (const T object)
{
  UserData userData;

  typename Map::iterator it = m_existMap.find(object);
  if (m_existMap.end() != it)
  {
    m_deletedMap[object] = it->second;
    userData = it->second;

    m_existMap.erase(it);
    logTrace ("Unregistered in %s scope: object = %p %s", m_id.c_str(), object, toString (userData).c_str());
  }
  else
  {
    typename Map::iterator it = m_deletedMap.find(object);
    if (it != m_deletedMap.end ())
    {
      debugError ("Double deallocation in %s: object = %p %s", m_id.c_str(), object, toString (it->second).c_str());
      debugAssert (false);
    }
    else
    {
      debugError ("Not found in %s: object = %p", m_id.c_str(), object);
      debugAssert (false);
    }
  }
  return userData;
}

template<typename T, typename UserData>
void AllocationList<T, UserData>::checkOnDelete()
{
  if (m_existMap.size() > 0)
  {
    debugError ("Memleak: not deallocated matrices");
    for (typename Map::iterator it = m_existMap.begin(); it != m_existMap.end(); ++it)
    {
      debug("Memleak in %s: object = %p %s not deallocated", m_id.c_str(), it->first, toString (it->second).c_str());
    }
    //debugAssert (false);
  }
}
}

#endif
