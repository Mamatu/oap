/*
 * Copyright 2016 - 2018 Marcin Matula
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
#include <string>
#include <functional>

#include "Routine.h"
#include "PatternsClassification.h"

int main(int argc, char** argv)
{
  using namespace std::placeholders;

  using RoutinesMap = std::map<std::string, oap::Routine*>;
  RoutinesMap routines;

  routines["patterns_classification"] = new oap::PatternsClassification ();

  auto getNamesList = [](const RoutinesMap& routines) -> std::string
  {
    std::vector<std::string> list;
    std::string listStr;
    for (auto it = routines.cbegin(); it != routines.cend(); ++it)
    {
      list.push_back (it->first);
    }

    for (size_t idx = 0; idx < list.size(); ++idx)
    {
      listStr += list[idx];
      if (idx < list.size() - 1)
      {
        listStr += ", ";
      }
    }
    return listStr;
  };

  auto nameCallback = [&routines, &getNamesList, argc, argv](const std::string& value)
  {
    auto it = routines.find (value);
    if (it == routines.end ())
    {
      std::stringstream sstream;
      sstream << "Routine \"" << value << "\" doesn\'t exist. Routines: ";
      sstream << getNamesList (routines) << '.';
      throw std::runtime_error (sstream.str ());
    }

    it->second->run (argc, argv);
  };

  oap::ArgsParser argParser;
  argParser.registerArg ("name", nameCallback);

  try
  {
    argParser.parse (argc, argv);
  }
  catch (const std::exception& exception)
  {
    logError ("%s", exception.what ());
    return 1;
  }

  return 0;
}
