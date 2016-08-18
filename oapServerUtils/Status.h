
#ifndef CODES_H
#define	CODES_H


namespace core {

    enum Status {
        STATUS_OK,
        STATUS_ERROR,
        STATUS_INVALID_HANDLE,
        STATUS_INVALID_ARGUMENT,
        STATUS_PROCESS_NOT_EXIST,
        STATUS_PROCESS_EXIST,
    };

    const char* getStr(Status code);
};


#endif	/* CODES_H */

