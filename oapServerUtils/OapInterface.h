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




#ifndef APIINTERACE_H
#define	APIINTERACE_H

#include "DebugLogs.h"
#include "Status.h"
#include "Argument.h"
#include "Writer.h"
#include "Reader.h"
#include "Callbacks.h"
#include "WrapperInterfaces.h"
#include "Loader.h"

#define API_SERVER_DEFAULT_PORT 4711

namespace core {
    class OapInterface : public utils::CallbacksManager {
    public:
        OapInterface();
        OapInterface(bool createCallbacks);
        OapInterface(utils::Callbacks* callbacks);
        OapInterface(utils::CallbacksManager* callbacksManager);
        virtual ~OapInterface();
        virtual Status registerFunction(utils::OapFunction* function) = 0;
        virtual Status executeFunction(uint connectionID, const char** names, uint namesCount, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) = 0;
        Status executeFunction(uint connectionID, const char* functionName, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output);
        Status executeFunction(uint connectionID, utils::Identificator* identificator, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output);
    };
}

#endif	/* MAINAPI_H */
