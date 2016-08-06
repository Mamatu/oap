/* 
 * File:   ObjectInfoImpl.cpp
 * Author: mmatula
 * 
 * Created on September 15, 2013, 3:31 PM
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
