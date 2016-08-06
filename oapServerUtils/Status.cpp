#include "Status.h"

namespace core {
    
    const char* getStr(Status code) {
        switch (code) {
            case STATUS_OK:
                return "STATUS: OK";
                break;

            case STATUS_ERROR:
                return "STATUS: ERROR";
                break;

            case STATUS_INVALID_ARGUMENT:
                return "STATUS: INVALID_ARGUMENT";
                break;

            case STATUS_INVALID_HANDLE:
                return "STATUS: INVALID_HANDLE";
                break;
        };
    }
}