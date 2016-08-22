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




#include "Argument.h"
#include "ArrayTools.h"
#include "Writer.h"
#include "Reader.h"
#include "DynamicLoader.h"
#include  <dlfcn.h>
#include <vector>
#include <algorithm>
namespace utils {

    class ModuleHandle;

    typedef std::pair<ModuleHandle*, std::vector<LHandle>::iterator > Pair;

    class ModuleHandle {
    public:

        ModuleHandle(const char* _path, void* _handle) : path(_path), handle(_handle) {
        }
        std::string path;
        void* handle;

        static std::pair<ModuleHandle*, std::vector<LHandle>::iterator> Convert(LHandle handle, std::vector<LHandle>& moduleHandles) {
            std::vector<LHandle>::iterator it = std::find(moduleHandles.begin(), moduleHandles.end(), handle);
            if (it != moduleHandles.end()) {
                ModuleHandle* moduleHandle = reinterpret_cast<ModuleHandle*> (handle.getPtr());
                return Pair(moduleHandle, it);
            }
            return Pair(NULL, moduleHandles.end());
        }
    };

    void* LoadedSymbol::getPtr() const {
        return this->ptr;
    }

    const char* LoadedSymbol::getFunctionName() const {
        return this->functionName.c_str();
    }

    LoadedSymbol::LoadedSymbol(const char* _functionName, void* _ptr) : functionName(_functionName), ptr(_ptr) {
    }

    LoadedSymbol::LoadedSymbol(const LoadedSymbol& orig) : functionName(orig.functionName), ptr(orig.ptr) {
    }

    LoadedSymbol::~LoadedSymbol() {
    }

    DynamicLoader::DynamicLoader(bool _keepHandles) : keepHandles(_keepHandles) {
    }

    DynamicLoader::~DynamicLoader() {
        if (this->keepHandles) {
            for (uint fa = 0; fa<this->moduleHandles.size(); fa++) {
                this->unload(moduleHandles[fa]);
            }
        }
    }

    LHandle DynamicLoader::load(const char* path) {
        void* handle = dlopen(path, RTLD_LAZY);
        if (handle) {
            ModuleHandle* moduleHandle = new ModuleHandle(path, handle);
            LHandle handle1 = LHandle(moduleHandle);
            this->moduleHandles.push_back(handle1);
            return handle1;
        }
        char* msg = dlerror();
        debug1(stderr, "Loader problem: %s \n", msg);
        return 0;
    }

    void DynamicLoader::unload(LHandle handle) {
        Pair pair = ModuleHandle::Convert(handle, this->moduleHandles);
        dlclose(pair.first->handle);
        delete pair.first;
        this->moduleHandles.erase(pair.second);
    }

    void DynamicLoader::execute(LHandle handle, const LoadedSymbol* loadedSymbol) {
        Pair pair = ModuleHandle::Convert(handle, this->moduleHandles);
        if (pair.first != NULL) {
            void* fhandle = pair.first->handle;
            const char* functionName = loadedSymbol->functionName.c_str();
            void* functionHandle = dlsym(fhandle, functionName);
            if (functionHandle != NULL) {
                this->invokeExecute(functionHandle, loadedSymbol);
            } else {
                char* msg = dlerror();
                debug1(stderr, "Loader problem: %s \n", msg);
                //free(msg);
            }
        }
    }
}
