/* 
 * File:   Serializable.h
 * Author: mmatula
 *
 * Created on February 18, 2014, 8:51 PM
 */

#ifndef SERIALIZABLE_H
#define	SERIALIZABLE_H

namespace utils {
    class Writer;
    class Reader;
    class Serializable {
    public:
        virtual void serialize(Writer& writer) = 0;
        virtual void deserialize(Reader& reader) = 0;
    };
}

#endif	/* SERIALIZABLE_H */

