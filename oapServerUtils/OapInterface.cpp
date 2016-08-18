
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