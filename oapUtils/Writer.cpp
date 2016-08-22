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



#include "Writer.h"
#include "Buffer.h"
#include "ArrayTools.h"
#include "Reader.h"
namespace utils {

    Writer::Writer() : buffer(NULL), size(0) {
    }

    Writer::Writer(const char* buffer, unsigned int size) {
        ArrayTools::set(&(this->buffer), this->size, buffer, size);
    }

    Writer::Writer(const Writer& writer) {
        ArrayTools::set(&(this->buffer), this->size, writer.buffer, writer.size);
    }

    Writer::~Writer() {
        if (buffer) {
            delete[] buffer;
        }
        this->size = 0;
    }

    void Writer::write(const std::string& text) {
        this->write(text.c_str(), text.length());
    }

    void Writer::write(const char* text) {
        this->write(text, strlen(text) * sizeof (char), typeid (text));
    }

    void Writer::write(LHandle handle) {
        size_t handleSize = sizeof (LHandle);
        this->write<size_t>(handleSize);
        this->write(&handle, handleSize);
    }

    void Writer::write(Serializable* serializable) {
        Writer writer;
        serializable->serialize(writer);
        int length = writer.getRealSize();
        char* bytes = NULL;
        writer.getBufferPtr(&bytes);
        this->write(bytes, length, typeid (Serializable*));
    }

    bool Writer::write(const char* buffer, unsigned int size, const std::type_info& typeInfo) {
        ArrayTools::add(&(this->buffer), this->size, buffer, size);
        return true;
    }

    void Writer::serialize(Writer& writer) {
        writer.setBuffer(this->buffer, this->size);
    }

    void Writer::deserialize(Reader& reader) {
        if (this->buffer != NULL) {
            delete[] (this->buffer);
        }
        reader.getBufferCopy(&(this->buffer), this->size);
    }

  unsigned  int Writer::getSize() const {
        return size;
    }

  unsigned  int Writer::getRealSize() const {
        return this->size;
    }

    void Writer::getBufferBytes(char* bytes, int offset, int size) const {
        if (this->size < size + offset) {
            return;
        }
        memcpy(bytes, this->buffer + offset, size);
    }

    void Writer::setBufferPtr(char* bytes,unsigned int size) {
        if (this->buffer != NULL) {
            delete[] this->buffer;
        }
        this->buffer = bytes;
        this->size = size;
    }

    void Writer::getBufferPtr(char** bytes) const {
        if (bytes != NULL) {
            (*bytes) = this->buffer;
        }
    }
}
