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

template<typename Object, typename UserData>
class AllocationList
{
  public:
    using InfosMap = std::unordered_map<const Object*, UserData>;

    AllocationList (const std::string& id);

    AllocationList (const AllocationList&) = delete;
    AllocationList (AllocationList&&) = delete;
    AllocationList& operator= (const AllocationList&) = delete;
    AllocationList& operator= (AllocationList&&) = delete;

    virtual ~AllocationList ();

    const InfosMap& getAllocated() const;

    void add (Object* matrix, const UserData& minfo);

    UserData remove (const Object* object);

    UserData getInfo (const Object* object) const;

		bool contains (const Object* object) const;

  private:
    std::string m_id;

    InfosMap m_existMap;
    InfosMap m_deletedMap;

    void checkOnDelete();
};
}

#include "oapAllocationList_impl.hpp"
#endif

