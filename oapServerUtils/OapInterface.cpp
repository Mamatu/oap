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




#include "OapInterface.h"
#include "ArrayTools.h"


namespace core {

    OapInterface::OapInterface() : utils::CallbacksManager() {
    }

    OapInterface::OapInterface(bool createCallbacks) : utils::CallbacksManager(createCallbacks) {
    }

    OapInterface::OapInterface(utils::Callbacks* callbacks) : utils::CallbacksManager(callbacks) {
    }

    OapInterface::OapInterface(utils::CallbacksManager* callbacksManager) : utils::CallbacksManager(callbacksManager) {
    }

    Status OapInterface::executeFunction(uint connectionID, utils::Identificator* identificator, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
        const char** names = NULL;
        int size = 0;
        utils::OapObject::ConvertToStrings(identificator->getRoot(), &names, size);
        ArrayTools::add(&names, size, identificator->getName());
        this->executeFunction(connectionID, names, size, input, args, argc, output);
        delete[] names;
    }

    Status OapInterface::executeFunction(uint connectionID, const char* functionName, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
        const char* names[1];
        int size = 1;
        names[0] = functionName;
        this->executeFunction(connectionID, names, size, input, args, argc, output);
    }

    OapInterface::~OapInterface() {
    }
}
