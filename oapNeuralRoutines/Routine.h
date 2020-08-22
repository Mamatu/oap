/*
 * Copyright 2016 - 2021 Marcin Matula
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

#ifndef OAP_ROUTINE_H
#define OAP_ROUTINE_H

#include "ArgsParser.h"

namespace oap
{

class Routine
{
  public:
    Routine() {}
    virtual ~Routine() {}

    Routine(const Routine&) = delete;
    Routine(Routine&&) = delete;
    Routine& operator=(const Routine&) = delete;
    Routine& operator=(Routine&&) = delete;

    int run (int argc, char* const* argv)
    {
      const oap::IArgsParser& parser = getArgsParser ();
      parser.parse (argc, argv);
      return runRoutine ();
    }

    virtual void onInterrupt() = 0;

  protected:
    virtual const oap::IArgsParser& getArgsParser() const = 0;
    virtual int runRoutine () = 0;
};;
}
#endif
