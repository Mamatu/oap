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




#ifndef BACKTRACEUTILS_H
#define	BACKTRACEUTILS_H

#include <string>
#include <vector>

class BacktraceUtils {
public:
    static BacktraceUtils& GetInstance();
    const std::vector<std::string>& readBacktrace(bool showThisObjectFrames = false);
    void printBacktrace(bool showThisObjectFrames = false);
protected:
    BacktraceUtils();
    BacktraceUtils(const BacktraceUtils& orig);
    virtual ~BacktraceUtils();
private:
    size_t m_size;
    void** m_buffer;
    int m_ptrsCount;
    char** m_strings;

    typedef std::vector<std::string> Backtrace;
    Backtrace m_backtrace;
    static BacktraceUtils* m_backtraceUtils;

    char** repeatReadBacktrace();
};

#endif	/* BACKTRACEUTILS_H */
