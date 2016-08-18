
#ifndef SERVERINTERFACES_H
#define	SERVERINTERFACES_H

#include "WrapperInterfaces.h"

namespace core {

    class Executable {
    public:
        virtual void execute() = 0;
    };

    template<class T> class Creator {
    private:
        utils::sync::Mutex mutex;
    protected:
        virtual T* newInstance() = 0;
        virtual void deleteInstance(T* object) = 0;
        virtual ~Creator();
    public:
        Creator();
        T* create();
        void destroy(T* obj);
    };

    template<class T> Creator<T>::Creator() {
    }

    template<class T> Creator<T>::~Creator() {
    }

    template<class T> T* Creator<T>::create() {
        mutex.lock();
        T* instance = this->newInstance();
        mutex.unlock();
        return instance;
    }

    template<class T> void Creator<T>::destroy(T* obj) {
        mutex.lock();
        this->destroy(obj);
        mutex.unlock();
    }

}
#endif	/* INTERFACES_H */

