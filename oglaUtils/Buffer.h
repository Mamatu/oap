/* 
 * File:   Interfaces.h
 * Author: marcin
 *
 * Created on 14 January 2013, 20:57
 */

#ifndef OGLA_BUFFER_H
#define	OGLA_BUFFER_H

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
        void getBuffer(char** bytes, unsigned int& size) const;
        void getBuffer(char** bytes) const;
        void extendBuffer(const char* bytes, unsigned int size);
        void extendBuffer(Buffer* buffer);
        virtual unsigned int getSize() const = 0;
    };
}

#endif	/* INTERFACES_H */

