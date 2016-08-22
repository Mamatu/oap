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




#ifndef READER_H
#define	READER_H

#include <string.h>
#include <stdio.h>
#include "Buffer.h"
#include "LHandle.h"

namespace utils {

    class Writer;

    class Reader : public Serializable, public Buffer {
        char* tempStrBuffer;
        int tempStrBufferSize;
    protected:
        int getRealSize() const;
        void setBufferPtr(char* bytes, unsigned int size);
        void getBufferPtr(char** bytes) const;
    public:
        Reader();
        Reader(const char* bytes, unsigned int size);
        Reader(const Reader& orig);
        Reader(const Writer& writer);
        virtual ~Reader();
        template<typename T> void read(T* t);
        template<typename T> void read(T* t, int length);
        int readInt();
        LHandle readHandle();
        char* readStr();
        void read(Serializable* serializable);
        void read(std::string& text);
        void getBufferBytes(char* bytes, int offset, int size) const;
        void serialize(Writer& writer);
        void deserialize(Reader& reader);
        unsigned int getSize() const;
        int getPosition() const;
        bool setPosition(int pos);
    protected:
        virtual bool read(char* buffer, int size);
        int position;
    private:
        unsigned int size;
        char* buffer;
        friend class Writer;
    };

    template<typename T> void Reader::read(T* t) {
        read((char*) t, sizeof (T));
    }

    template<typename T> void Reader::read(T* t, int length) {
        read((char*) t, sizeof (T) * length);
    }
}
#endif
