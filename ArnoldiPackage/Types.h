#ifndef TYPES_H
#define	TYPES_H

enum Status {
    STATUS_OK,
    STATUS_ERROR,
    STATUS_INVALID_PARAMS,
    STATUS_NO_OUTPUTS_DEFINED
};

class Buffer {
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

#endif	/* TYPES_H */

