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

#ifndef OAP_ALLOCATION_LIST_H
#define OAP_ALLOCATION_LIST_H

#include <unordered_map>

namespace oap
{

template<typename T, typename UserData>
class AllocationList
{
  public:
    using Map = std::unordered_map<T, UserData>;

    AllocationList (const std::string& id);

    AllocationList (const AllocationList&) = delete;
    AllocationList (AllocationList&&) = delete;
    AllocationList& operator= (const AllocationList&) = delete;
    AllocationList& operator= (AllocationList&&) = delete;

    virtual ~AllocationList ();

    const Map& getAllocated() const;

    void add (const T object, const UserData& minfo);

    UserData remove (const T object);

    UserData getUserData (const T object) const;

    virtual std::string toString (const UserData&) const = 0;

		bool contains (const T object) const;

  private:
    std::string m_id;

    Map m_existMap;
    Map m_deletedMap;

    void checkOnDelete();
};
}

#include "oapAllocationList_impl.hpp"
#endif

