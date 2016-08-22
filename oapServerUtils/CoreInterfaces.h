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
