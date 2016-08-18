
#include <map>
#include "Module.h"

namespace utils {

    Module::Module() : message("") {
    }

    Module::Module(const Module& orig) {
        this->message = orig.message;
    }

    Module::~Module() {
    }

    bool Module::isError() const {
        return this->message.length() > 0;
    }

    bool Module::isOk() const {
        return !isError();
    }

    void Module::printMessage(FILE* file) {
        if (message.length() > 0) {
            fprintf(file, "Msg : %s \n", message.c_str());
        }
        message.clear();
    }

    void Module::getMessage(std::string& str) {
        str = this->message;
        this->message.clear();
    }

    void Module::addMessageLine(const char* msg) {
        this->message += msg;
        this->message += "\n";
    }
}