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




#ifndef OBJECTINFO_H
#define	OBJECTINFO_H

#include "FunctionInfo.h"

namespace utils {

    class ObjectInfo {
    protected:
        ObjectInfo();
        virtual ~ObjectInfo();
    public:
        virtual const char* getName() const = 0;
        virtual utils::FunctionInfo* getFunctionInfo(uint index) const = 0;
        virtual uint getFunctionsCount() const = 0;
        virtual utils::ObjectInfo* getObjectInfo(uint index) const = 0;
        virtual uint getObjectsCount() const = 0;
    };
}

#endif	/* OBJECTINFO_H */
