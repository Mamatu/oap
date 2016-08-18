
#include <stdio.h>
#include <fstream>

#include "Reader.h"
#include "Writer.h"
#include "ArrayTools.h"

namespace utils {

    Reader::Reader() : buffer(NULL), size(0), position(0), tempStrBuffer(NULL), tempStrBufferSize(0) {
    }

    Reader::Reader(const char* buffer, unsigned int size) : position(0), tempStrBuffer(NULL), tempStrBufferSize(0) {
        this->buffer = NULL;
        this->size = 0;
        ArrayTools::set(&(this->buffer), this->size, buffer, size);
    }

    Reader::Reader(const Reader& reader) : position(0), tempStrBuffer(NULL), tempStrBufferSize(0) {
        this->buffer = NULL;
        this->size = 0;
        ArrayTools::set(&(this->buffer), this->size, reader.buffer, reader.size);
    }

    Reader::Reader(const Writer& writer) : position(0), tempStrBuffer(NULL), tempStrBufferSize(0) {
        this->buffer = NULL;
        this->size = 0;
        ArrayTools::set(&(this->buffer), this->size, writer.buffer, writer.size);
    }

    Reader::~Reader() {
        if (this->buffer) {
            delete[] buffer;
        }
        if (this->tempStrBuffer != NULL) {
            delete[] this->tempStrBuffer;
        }
        this->size = 0;
    }

    void Reader::getBufferBytes(char* bytes, int offset, int size) const {
        if (this->size < size + offset) {
            return;
        }
        memcpy(bytes, this->buffer + offset, size);
    }

    int Reader::getRealSize() const {
        return this->size;
    }

    void Reader::setBufferPtr(char* bytes, unsigned int size) {
        if (this->buffer != NULL) {
            delete[] this->buffer;
        }
        this->buffer = bytes;
        this->size = size;
    }

    void Reader::getBufferPtr(char** bytes) const {
        if (bytes != NULL) {
            (*bytes) = this->buffer;
        }
    }

    LHandle Reader::readHandle() {
        int size = this->readInt();
        LHandle handle = 0;
        this->read(&handle, size);
        return handle;
    }

    int Reader::readInt() {
        int output = 0;
        this->read(&output);
        return output;
    }

    char* Reader::readStr() {
        std::string text;
        this->read(text);
        char* out = new char[text.length() + 1];
        out[text.length()] = 0;
        memcpy(out, text.c_str(), text.length());
        return out;
    }

    void Reader::read(Serializable* serializable) {
        int bytesSize = 0;
        this->read(&bytesSize);
        char* bytes = new char[bytesSize];
        this->read(bytes, bytesSize);
        Reader reader(bytes, bytesSize);
        serializable->deserialize(reader);
        delete[] bytes;
    }

    void Reader::read(std::string& text) {
        int size;
        this->read(&size);
        if (size > tempStrBufferSize) {
            if (tempStrBuffer != NULL) {
                delete[] tempStrBuffer;
            }
            tempStrBuffer = new char[size];
            tempStrBufferSize = size;
        }
        this->read(tempStrBuffer, tempStrBufferSize);
        text = std::string(tempStrBuffer, tempStrBufferSize);
    }

    bool Reader::read(char* buffer, int size) {
        if (this->size - this->position < size) {
            return false;
        }
        memcpy(buffer, this->buffer + position, size);
        position += size;
        return true;
    }

    void Reader::serialize(Writer& writer) {
        writer.setBuffer(this->buffer, this->size);
    }

    void Reader::deserialize(Reader& reader) {
        if (this->buffer != NULL) {
            delete[] (this->buffer);
        }
        reader.getBufferCopy(&(this->buffer), this->size);
    }

    unsigned int Reader::getSize() const {
        return size;
    }

    int Reader::getPosition() const {
        return position;
    }

    bool Reader::setPosition(int pos) {
        if (pos >= 0 && pos <= this->size) {
            this->position = pos;
            return true;
        }
        return false;
    }

}