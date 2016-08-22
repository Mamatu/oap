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




#ifndef PARAMETERS_H
#define	PARAMETERS_H

#include "DebugLogs.h"
#include "Math.h"

class Parameters {
public:

    template <typename T>class Parameter {
    public:

        Parameter() : isAvailable(false) {
        }
        T value;
        bool isAvailable;
    };

    Parameters(bool isThreadsCount = true, bool isSerieLimit = false);
    virtual ~Parameters();

    bool isParamThreadsCount() const;
    void setThreadsCount(uintt count);
    uintt getThreadsCount() const;

    bool isParamSerieLimit() const;
    void setSerieLimit(uintt limit);
    uintt getSerieLimit() const;
private:
    Parameter<uintt> threadsCount;
    Parameter<uintt> serieLimit;
    Parameters(const Parameters& orig);
};

#endif	/* PARAMETERS_H */
