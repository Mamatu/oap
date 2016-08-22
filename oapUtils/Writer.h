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




#ifndef WRITER_H
#define	WRITER_H

#include <string>
#include <string.h>
#include <stdio.h>
#include "ThreadUtils.h"
#include "Buffer.h"
#include <typeinfo>

namespace utils {

    class Reader;

    class Writer : public Serializable, public Buffer {
    protected:
        unsigned int getRealSize() const;
        void setBufferPtr(char* bytes, unsigned int size);
        void getBufferPtr(char** bytes) const;
    public:
        Writer();
        Writer(const char* buffer, unsigned int size);
        Writer(const Writer& writer);
        virtual ~Writer();
        template<typename T> void write(T t);
        template<typename T> void write(T* t, int length);
        void write(LHandle handle);
        void write(const std::string& text);
        void write(const char* text);
        void write(Serializable* serializable);
        void getBufferBytes(char* bytes, int offset, int size) const;
        void serialize(Writer& writer);
        void deserialize(Reader& reader);
        unsigned int getSize() const;
    protected:
        virtual bool write(const char* buffer, unsigned int size, const std::type_info& typeInfo);
    private:
        char* buffer;
        unsigned int size;
        friend class Reader;
    };

    template<typename T> void Writer::write(T t) {
        this->write((const char*) &t, sizeof (T), typeid (t));
    }

    template<typename T> void Writer::write(T* t, int length) {
        this->write((const char*) t, length * sizeof (T), typeid (t));
    }
}

#endif	/* WRITER_H */
