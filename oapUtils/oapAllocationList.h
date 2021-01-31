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

#ifndef OAP_ALLOCATION_LIST_H
#define OAP_ALLOCATION_LIST_H

#include <unordered_map>
#include <map>
#include <functional>

namespace oap
{

template<typename T, typename UserData, typename ToString>
class AllocationList
{
  public:
    using Map = std::map<T, UserData>;

    AllocationList (const std::string& id, ToString&& to_string);

    AllocationList (const AllocationList&) = delete;
    AllocationList (AllocationList&&) = delete;
    AllocationList& operator= (const AllocationList&) = delete;
    AllocationList& operator= (AllocationList&&) = delete;

    virtual ~AllocationList ();

    const Map& getAllocated() const;

    void add (const T object, const UserData& minfo);

    UserData remove (const T object);

    UserData getUserData (const T object) const;

		bool contains (const T object) const;

  private:
    std::string m_id;

    ToString to_string;

    Map m_existMap;
    Map m_deletedMap;

  protected:
    void checkOnDelete();
};
}

#include "oapAllocationList_impl.hpp"
#endif

