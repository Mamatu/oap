
#ifndef OAP_MODULE_H
#define	OAP_MODULE_H
#include <queue>
#include <map>
#include <string>
#include "DebugLogs.h"
#include <stdio.h>

#define CONTROL_CRITICALS() //if(this->isCriticalError()){return;}

#ifdef DEBUG
#define OAP_CHECK_ERROR(module) if(module.isError()){ module.printMessage(stderr); } 
#else
#define OAP_CHECK_ERROR(module)
#endif

namespace utils {

    class Module {
    public:
        Module();
        Module(const Module& orig);
        virtual ~Module();
        bool isError() const;
        bool isOk() const;
        void printMessage(FILE* s = stdout);
        void getMessage(std::string& msg);
    protected:
        void addMessageLine(const char* msg);
    private:
        std::string message;
    };
}

#endif	/* ERRORSTACK_H */

