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




#ifndef ARGUMENTWRITER_H
#define	ARGUMENTWRITER_H

#include "Argument.h"
#include "Writer.h"


namespace utils {

    class ArgumentsWriter : public utils::Writer {
    public:
        ArgumentsWriter();
        virtual ~ArgumentsWriter();
        utils::ArgumentType* getArgumentsTypes(int& length);
    protected:
        virtual bool write(const char* buffer, int size, std::type_info& typeInfo);
    private:
        std::vector<utils::ArgumentType> argumentsTypes;
    };
}
#endif	/* ARGUMENTWRITER_H */
