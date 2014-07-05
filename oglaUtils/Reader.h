/* 
 * File:   ByteReader.h
 * Author: marcin
 *
 * Created on 16 luty 2012, 20:31
 */

#ifndef READER_H
#define	READER_H

#include <string.h>
#include <stdio.h>
#include "Buffer.h"

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

