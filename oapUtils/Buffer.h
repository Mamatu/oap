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




#ifndef OAP_BUFFER_H
#define	OAP_BUFFER_H

#include "DebugLogs.h"
#include "Argument.h"
#include "Serializable.h"

namespace utils {

    class Buffer {
    protected:
        virtual void setBufferPtr(char* bytes, unsigned int size) = 0;
        virtual void getBufferPtr(char** bytes) const = 0;
    public:
        virtual void getBufferBytes(char* bytes, int offset, int size) const = 0;

        void getBufferBytes(char* bytes, unsigned int size) const;
        void setBuffer(const char* bytes, unsigned int size);
        void setBuffer(Buffer* buffer);
        void getBufferCopy(char** bytes, unsigned int& size) const;
        void getBuffer(char** bytes) const;
        void extendBuffer(const char* bytes, unsigned int size);
        void extendBuffer(Buffer* buffer);
        virtual unsigned int getSize() const = 0;
    };
}

#endif	/* INTERFACES_H */
