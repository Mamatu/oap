/*
 * Copyright 2016, 2017 Marcin Matula
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
