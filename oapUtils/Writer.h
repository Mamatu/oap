
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

