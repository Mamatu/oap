/* 
 * File:   Argument.h
 * Author: marcin
 *
 * Created on 03 January 2013, 22:56
 */

#ifndef DYNAMICLOADER_H
#define	DYNAMICLOADER_H

#include <string.h>
#include <string>
#include <vector>
#include "Types.h"

namespace utils {

    class DynamicLoader;

    class LoadedSymbol {
        std::string functionName;
        void* ptr;
    public:

        void* getPtr() const;
        const char* getFunctionName() const;

        LoadedSymbol(const char* functionName, void* ptr);
        LoadedSymbol(const LoadedSymbol& orig);
        ~LoadedSymbol();
        friend class DynamicLoader;
    };

    class DynamicLoader {
        std::vector<LHandle> moduleHandles;
        bool keepHandles;
    protected:
        virtual void invokeExecute(void* functionHandle, const LoadedSymbol* loadedSymbol) = 0;
    public:
        DynamicLoader(bool keepHandles = false);
        virtual ~DynamicLoader();
        LHandle load(const char* path);
        void unload(LHandle handle);
        void execute(LHandle handle, const LoadedSymbol* symbol);
    };
}
#endif	/* ARGUMENT_H */

