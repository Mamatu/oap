/* 
 * File:   FunctionInfo.h
 * Author: mmatula
 *
 * Created on September 15, 2013, 2:41 PM
 */

#ifndef FUNCTIONINFO_H
#define	FUNCTIONINFO_H

#include <vector>
#include <map>

#include "Socket.h"
#include "Reader.h"
#include "Writer.h"
#include "Argument.h"
#include "WrapperInterfaces.h"
#include "ThreadUtils.h"
#include <vector>
#include <map>

#include "Socket.h"
#include "Reader.h"
#include "Writer.h"
#include "Callbacks.h"
#include "Argument.h"
#include "WrapperInterfaces.h"
#include "ThreadUtils.h"
#include <string.h>

namespace utils {

    class FunctionInfo : public utils::Serializable {
    protected:
        FunctionInfo();
        virtual ~FunctionInfo();
    public:
        virtual LHandle getConnectionID() const = 0;
        virtual void getName(char*& name, uint index) const = 0;
        virtual int getNamesCount() const = 0;
        virtual utils::ArgumentType getInputArgumentType(uint index) const = 0;
        virtual uint getInputArgc() const = 0;
        virtual utils::ArgumentType getOutputArgumentType(uint index) const = 0;
        virtual uint getOutputArgc() const = 0;
    };
}

#endif	/* FUNCTIONINFO_H */

