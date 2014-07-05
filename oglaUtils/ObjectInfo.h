/* 
 * File:   ObjectInfo.h
 * Author: mmatula
 *
 * Created on September 15, 2013, 3:31 PM
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

