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
