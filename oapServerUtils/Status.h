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




#ifndef CODES_H
#define	CODES_H


namespace core {

    enum Status {
        STATUS_OK,
        STATUS_ERROR,
        STATUS_INVALID_HANDLE,
        STATUS_INVALID_ARGUMENT,
        STATUS_PROCESS_NOT_EXIST,
        STATUS_PROCESS_EXIST,
    };

    const char* getStr(Status code);
};


#endif	/* CODES_H */
