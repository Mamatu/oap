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




#include "ObjectInfoImpl.h"
#include "FunctionInfoImpl.h"
namespace utils {

    ObjectInfoImpl::ObjectInfoImpl(utils::OapObject* object) {
        this->name = object->getName();
        for (uint fa = 0; fa < object->getFunctionsContainer()->getCount(); fa++) {
            this->functions.push_back(new utils::FunctionInfoImpl(object->getFunctionsContainer()->get(fa)));
        }
        for (uint fa = 0; fa < object->getObjectsContainer()->getCount(); fa++) {
            this->objects.push_back(new utils::ObjectInfoImpl(object->getObjectsContainer()->get(fa)));
        }
    }

    ObjectInfoImpl::~ObjectInfoImpl() {
        for (uint fa = 0; fa < this->functions.size(); fa++) {
            delete this->functions[fa];
        }
        for (uint fa = 0; fa < this->objects.size(); fa++) {
            delete this->objects[fa];
        }
    }

    const char* ObjectInfoImpl::getName() const {
        return name.c_str();
    }

    utils::FunctionInfo* ObjectInfoImpl::getFunctionInfo(uint index) const {
        return functions[index];
    }

    uint ObjectInfoImpl::getFunctionsCount() const {
        return functions.size();
    }

    utils::ObjectInfo* ObjectInfoImpl::getObjectInfo(uint index) const {
        return objects[index];
    }

    uint ObjectInfoImpl::getObjectsCount() const {
        return objects.size();
    }

}
