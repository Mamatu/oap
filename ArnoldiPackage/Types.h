#ifndef OGLA_ARNOLDI_PACKAGE_TYPES_H
#define	OGLA_ARNOLDI_PACKAGE_TYPES_H

#include "stdlib.h"

namespace api {

    enum Mode {
        /**
         * Stete can be modified only by user.
         */
        MODE_DIRECTLY,
        /*
         * State can be modified by user and implementation.
         */
        MODE_UNDIRECTLY
    };

    class Buffer {
    protected:
        virtual ~Buffer();
    public:
        /**
         * Buffer constructor sets data on NUL nad size on 0;
         */
        Buffer();
        /**
         * Pointer to data.
         */
        char* data;
        /**
         * Size in bytes data.
         */
        size_t size;
    };

    typedef unsigned int Handle;
}

#endif	/* TYPES_H */

