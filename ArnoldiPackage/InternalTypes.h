#ifndef INTERNALTYPES_H
#define	INTERNALTYPES_H

#include <stdio.h>


enum State {
    STATE_STARTED,
    STATE_STOPED
};

class Buffer {
public:
    char* data;
    size_t size;
};

#endif	/* INTERNALTYPES_H */

