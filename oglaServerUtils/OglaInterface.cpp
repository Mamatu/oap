/* 
 * File:   MainAPI.cpp
 * Author: marcin
 * 
 * Created on 30 April 2013, 22:08
 */

#include "OglaInterface.h"
#include "ArrayTools.h"


namespace core {

    OglaInterface::OglaInterface() : utils::CallbacksManager() {
    }

    OglaInterface::OglaInterface(bool createCallbacks) : utils::CallbacksManager(createCallbacks) {
    }

    OglaInterface::OglaInterface(utils::Callbacks* callbacks) : utils::CallbacksManager(callbacks) {
    }

    OglaInterface::OglaInterface(utils::CallbacksManager* callbacksManager) : utils::CallbacksManager(callbacksManager) {
    }

    Status OglaInterface::executeFunction(uint connectionID, utils::Identificator* identificator, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
        const char** names = NULL;
        int size = 0;
        utils::OglaObject::ConvertToStrings(identificator->getRoot(), &names, size);
        ArrayTools::add(&names, size, identificator->getName());
        this->executeFunction(connectionID, names, size, input, args, argc, output);
        delete[] names;
    }

    Status OglaInterface::executeFunction(uint connectionID, const char* functionName, utils::Writer& input, utils::ArgumentType* args, uint argc, utils::Reader& output) {
        const char* names[1];
        int size = 1;
        names[0] = functionName;
        this->executeFunction(connectionID, names, size, input, args, argc, output);
    }

    OglaInterface::~OglaInterface() {
    }
}