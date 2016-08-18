
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

