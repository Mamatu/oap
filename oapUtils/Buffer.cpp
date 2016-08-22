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



#include "Buffer.h"
#include "Writer.h"
#include "Reader.h"
#include <string.h>

namespace utils {

#define PUT(type,v)\
        this->put((const char*) &v, sizeof (type));

#define GET1(type,size)\
    char buffer[size];\
    this->get(buffer, size);\
    return *(type*)(buffer);

#define GET(type)\
    GET1(type,sizeof(type))    

    void Buffer::getBufferBytes(char* bytes, unsigned int size) const {
        this->getBufferBytes(bytes, 0, size);
    }

    void Buffer::setBuffer(const char* bytes, unsigned int size) {
        char* nbytes = new char[size];
        memcpy(nbytes, bytes, size);
        this->setBufferPtr(nbytes, size);
    }

    void Buffer::setBuffer(Buffer* buffer) {
        char* ptr = NULL;
        unsigned int size = buffer->getSize();
        buffer->getBufferPtr(&ptr);
        this->setBuffer(ptr, size);
    }

    void Buffer::getBufferCopy(char** bytes, unsigned int& size) const {
        if (bytes != NULL) {
            int realSize = this->getSize();
            (*bytes) = new char[realSize];
            this->getBufferBytes((*bytes), realSize);
            size = realSize;
        }
    }

    void Buffer::extendBuffer(const char* bytes, unsigned int size) {
        if (bytes != NULL && size != 0) {
            char* currentBuffer = NULL;
            int currentSize = 0;
            this->getBufferPtr(&currentBuffer);
            currentSize = this->getSize();
            int newSize = currentSize + size;
            char* nbytes = new char[newSize];
            memcpy(nbytes, currentBuffer, currentSize);
            memcpy(nbytes + currentSize, bytes, size);
            this->setBufferPtr(nbytes, newSize);
        }
    }

    void Buffer::extendBuffer(Buffer* buffer) {
        char* ptr = NULL;
        buffer->getBufferPtr(&ptr);
        this->extendBuffer(ptr, buffer->getSize());
    }

    void Buffer::getBuffer(char** bytes) const {
        unsigned int size = 0;
        this->getBufferCopy(bytes, size);
    }
}
