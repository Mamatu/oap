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




#ifndef DYNAMICLOADER_H
#define	DYNAMICLOADER_H

#include <string.h>
#include <string>
#include <vector>
#include "DebugLogs.h"

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
