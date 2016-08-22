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




#ifndef RANDOMS_H
#define	RANDOMS_H

#include "Async.h"


namespace utils {

    template<typename T> class Iterable {
    public:
        Iterable();
        virtual bool end() = 0;
        virtual void add(T t) = 0;
    protected:
        virtual ~Iterable();
    };

    template<typename T> Iterable<T>::Iterable() {
    }

    template<typename T> Iterable<T>::~Iterable() {
    }

    template<typename T> class NumbersAsyncGenerator : public AsyncProcess {
    public:
        typedef T(*Generate_f)(void* userPtr);

        NumbersAsyncGenerator(Generate_f generate = NULL, AsyncQueue* async = NULL, void* userPtr = NULL);
        virtual ~NumbersAsyncGenerator();

        int generateNumbers(T* buffer, uint size, Generate_f generate = NULL, void* userPtr = NULL);
        int generateNumbers(Iterable<T>* iterable, Generate_f generate = NULL, void* userPtr = NULL);

    protected:
        static void GenerateSet(void* ptr);
        static void GenerateSet1(void* ptr);
        Generate_f defaultFunc;
        void* defaultUserPtr;
    private:

        class Params {
        public:
            Params();
            T* buffer;
            void* userPtr;
            uint size;
            Generate_f generate;
            Iterable<T>* iterable;
        };
    };

    template<typename T> NumbersAsyncGenerator<T>::NumbersAsyncGenerator(Generate_f generate, AsyncQueue* async, void* userPtr) : AsyncProcess(async), defaultFunc(generate),
    defaultUserPtr(userPtr) {
    }

    template<typename T> NumbersAsyncGenerator<T>::~NumbersAsyncGenerator() {
    }

    template<typename T> void NumbersAsyncGenerator<T>::GenerateSet(void* ptr) {
        NumbersAsyncGenerator::Params* params = (NumbersAsyncGenerator::Params*) ptr;
        for (uint fa = 0; fa < params->size; fa++) {
            params->buffer[fa] = params->generate(params->userPtr);
        }
    }

    template<typename T> void NumbersAsyncGenerator<T>::GenerateSet1(void* ptr) {
        NumbersAsyncGenerator::Params* params = (NumbersAsyncGenerator::Params*) ptr;
        while (params->iterable->end() == false) {
            params->iterable->add(params->generate(params->userPtr));
        }
    }

    template<typename T> int NumbersAsyncGenerator<T>::generateNumbers(T* buffer, uint size, Generate_f generate, void* userPtr) {
        if (defaultFunc == NULL && generate == NULL) {
            return 1;
        }
        Params* params = new Params();
        params->buffer = buffer;
        params->size = size;
        params->generate = (generate == NULL) ? defaultFunc : generate;
        params->userPtr = (userPtr == NULL) ? defaultUserPtr : userPtr;
        if (params->iterable == NULL) {
            this->getInstance().execute(NumbersAsyncGenerator::GenerateSet, params);
        } else {
            this->getInstance().execute(NumbersAsyncGenerator::GenerateSet1, params);
        }
        return 0;
    }

    template<typename T> int NumbersAsyncGenerator<T>::generateNumbers(Iterable<T>* iterable, Generate_f generate, void* userPtr) {
        if (defaultFunc == NULL && generate == NULL) {
            return 1;
        }
        Params* params = new Params();
        params->iterable = iterable;
        params->generate = (generate == NULL) ? defaultFunc : generate;
        params->userPtr = (userPtr == NULL) ? defaultUserPtr : userPtr;
        if (params->iterable == NULL) {
            this->getInstance().execute(NumbersAsyncGenerator::GenerateSet, params);
        } else {
            this->getInstance().execute(NumbersAsyncGenerator::GenerateSet1, params);
        }
        return 0;
    }

    template<typename T> NumbersAsyncGenerator<T>::Params::Params() : iterable(NULL), buffer(NULL), size(0) {
    }
}
#endif	/* RANDOMS_H */
