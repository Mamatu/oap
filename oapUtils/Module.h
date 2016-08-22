/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */




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
