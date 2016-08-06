/* 
 * File:   Loader.h
 * Author: marcin
 *
 * Created on 25 April 2013, 22:09
 */

#ifndef LOADER_H
#define	LOADER_H

#include "DebugLogs.h"
#include "WrapperInterfaces.h"
#include "DynamicLoader.h"
#include <algorithm>
#include <vector>


namespace core {

class ExternalHandle {
public:
    uint count;
    std::vector<LHandle> internalHandles;
    LHandle dynamicHandle;
    void* ptr;
};

template<typename T> class Loader {
private:

    class DynamicLoaderImpl : public utils::DynamicLoader {
        typedef void (*GetCount)(uint& size);
        typedef void (*LoadImpls)(T** objects);
        typedef void (*UnloadImpl)(T* object);
        Loader<T>* loader;
    public:

        DynamicLoaderImpl(Loader<T>* _loader) : DynamicLoader(), loader(_loader) {
        }

        void invokeExecute(void* handle, const utils::LoadedSymbol* loadedSymbol) {
            const char* function = loadedSymbol->getFunctionName();
            void* ptr = loadedSymbol->getPtr();
            if (strcmp(function, loader->getLoadSymbolStr()) == 0) {
                ExternalHandle* externalHandle = (ExternalHandle*) ptr;
                LoadImpls loadImpls = (LoadImpls) handle;
                T** objects = new T*[externalHandle->count];
                memset(objects, 0, externalHandle->count * sizeof (T *));
                loadImpls(objects);
                for (uint fa = 0; fa < externalHandle->count; fa++) {
                    externalHandle->internalHandles.push_back(*(LHandle*) objects[fa]);
                }
                externalHandle->ptr = (void*) objects;
            } else if (strcmp(function, loader->getUnloadSymbolStr()) == 0) {
                ExternalHandle* externalHandle = (ExternalHandle*) ptr;
                UnloadImpl unloadImpl = (UnloadImpl) handle;
                for (uint fa = 0; fa < externalHandle->internalHandles.size(); fa++) {
                    unloadImpl((T*) externalHandle->internalHandles[fa].getPtr());
                }
            } else if (strcmp(function, loader->getCountStr()) == 0) {
                uint* countPtr = (uint*) ptr;
                GetCount getCount = (GetCount) handle;
                uint count = 0;
                getCount(count);
                (*countPtr) = count;
            }
        }
    };

    const char* getLoadSymbolStr() const {
        return loadSymbol.c_str();
    }

    const char* getUnloadSymbolStr() const {
        return unloadSymbol.c_str();
    }

    const char* getCountStr() const {
        return getCountSymbol.c_str();
    }
public:

    class Callback {
    public:
        virtual bool setCount(uint count) = 0;
        virtual bool setLoadedImpls(T** impls, uint handle) = 0;
        virtual void setUnloadedImpl(T* impl) = 0;
    };
    std::vector<Loader<T>::Callback*> callbacks;
private:

    bool invokeCallbacks(uint type, void* data) {
        typename std::vector<Loader<T>::Callback*>::iterator it = callbacks.begin();
        for (; it != callbacks.end(); it++) {
            uint count = 0;
            std::pair<T**, uint>* pair;
            T* impl = NULL;
            switch (type) {
                case 1:
                    count = *((uint*) data);
                    if ((*it)->setCount(count) == false) {
                        return false;
                    }
                    break;
                case 2:
                    pair = (std::pair<T**, uint>*) data;
                    if ((*it)->setLoadedImpls(pair->first, pair->second) == false) {
                        return false;
                    }
                    break;
                case 3:
                    impl = ((T*) data);
                    (*it)->setUnloadedImpl(impl);
                    break;
            };
        }
        return true;
    }

    typedef std::vector<LHandle> Vector;
    utils::DynamicLoader* dynamicLoader;
    std::vector<LHandle> externalHandles;
    std::vector<LHandle> internalHandles;
    std::string loadSymbol;
    std::string unloadSymbol;
    std::string getCountSymbol;
public:

    Loader(const char* _loadSymbol, const char* _unloadSymbol, const char* _getCountSymbol) :
    loadSymbol(_loadSymbol), unloadSymbol(_unloadSymbol), getCountSymbol(_getCountSymbol), dynamicLoader(new DynamicLoaderImpl(this)) {
    }

    virtual ~Loader() {
    }

    int load(const char* path, LHandle& rhandle) {
        LHandle dynamicHandle = dynamicLoader->load(path);
        uint count = 0;
        utils::LoadedSymbol loadComponentsCount(getCountSymbol.c_str(), &count);
        dynamicLoader->execute(dynamicHandle, &loadComponentsCount);
        if (this->invokeCallbacks(1, &count) == false) {
            return 0;
        }
        ExternalHandle* externalHandle = new ExternalHandle();
        externalHandle->count = count;
        utils::LoadedSymbol loadComponentsSymbol(loadSymbol.c_str(), externalHandle);
        dynamicLoader->execute(dynamicHandle, &loadComponentsSymbol);
        externalHandle->dynamicHandle = dynamicHandle;
        LHandle handle = LHandle(externalHandle);
        std::pair<T**, LHandle> pairModules((T**) externalHandle->ptr, handle);
        if (this->invokeCallbacks(2, &pairModules) == false) {
            delete externalHandle;
            return 0;
        }
        this->externalHandles.push_back(handle);
        rhandle = handle;
        return 0;
    }

    int remove(LHandle handle) {
        if (handle == 0) {
            return 1;
        }
        Vector::iterator it = std::find(this->internalHandles.begin(), this->internalHandles.end(), handle);
        if (it != this->internalHandles.end()) {
            this->internalHandles.erase(it);
            return -1;
        } else {
            utils::LoadedSymbol unloadComponentsSymbol(unloadSymbol.c_str(), &externalHandles);
            ExternalHandle* externalHandle = reinterpret_cast<ExternalHandle*> (handle.getPtr());
            this->externalHandles.erase(it);
            T** impls = (T**) externalHandle->ptr;
            for (uint fa = 0; fa < externalHandle->count; fa++) {
                this->invokeCallbacks(3, &impls[fa]);
            }
            this->dynamicLoader->unload(externalHandle->dynamicHandle);
            for (uint fa = 0; fa < externalHandle->internalHandles.size(); fa++) {
                this->remove(externalHandle->internalHandles[fa]);
            }
            dynamicLoader->execute(externalHandle->dynamicHandle, &unloadComponentsSymbol);
            delete externalHandle;
            return 0;
        }
    }
    friend class DynamicLoaderImpl;
};
}

#endif	/* LOADER_H */

