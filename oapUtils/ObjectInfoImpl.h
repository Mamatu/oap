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




#ifndef OBJECTINFOIMPL_H
#define	OBJECTINFOIMPL_H

#include "ObjectInfo.h"
#include "FunctionInfoImpl.h"
#include <vector>

namespace utils {
    class ObjectInfoImpl : public utils::ObjectInfo {
        std::vector<ObjectInfoImpl*> objects;
        std::vector<FunctionInfoImpl*> functions;
        std::string name;
    public:
        ObjectInfoImpl(utils::OapObject* object);
        virtual ~ObjectInfoImpl();
        const char* getName() const;
        utils::FunctionInfo* getFunctionInfo(uint index) const;
        uint getFunctionsCount() const;
        utils::ObjectInfo* getObjectInfo(uint index) const;
        uint getObjectsCount() const;
    };
}
#endif	/* OBJECTINFOIMPL_H */
