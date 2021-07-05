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

#include "ArgsParser.hpp"

#include "Logger.hpp"

namespace oap
{

ArgsParser::ArgsParser()
{}

bool ArgsParser::isInitialized (const ArgsParser::Arg& arg) const
{
  return !arg.first.empty ();
}

ArgsParser::Arg ArgsParser::getArg (const std::string& larg) const
{
  Arg rarg;
  
  auto check = [&rarg, &larg](const Arg& arg, const Callback&)
  {
    if (arg.first == larg)
    {
      rarg = arg;
      return false;
    }
    return true;
  };

  iterateArgs_while (check);
  return rarg;
}

ArgsParser::Arg ArgsParser::getArg (char sarg) const
{
  Arg rarg;
  
  auto check = [&rarg, &sarg](const Arg& arg, const Callback&)
  {
    if (arg.second == sarg)
    {
      rarg = arg;
      return false;
    }
    return true;
  };

  iterateArgs_while (check);
  return rarg;
}

bool ArgsParser::containsArg (const std::string& larg) const
{
  auto check = getArg (larg);
  return isInitialized (check);
}

bool ArgsParser::containsArg (char sarg) const
{
  auto check = getArg (sarg);
  return isInitialized (check);
}

void ArgsParser::registerArg (const std::string& larg, const ArgsParser::Callback& callback)
{
  throwsIfContains<const std::string&> (larg);

  m_callbacks[std::make_pair(larg, 0)] = callback;
}

void ArgsParser::parse (int argc, char* const* argv) const
{
  if (m_callbacks.size() > 0 && argc == 1)
  {
    std::stringstream sstream;
    sstream << "Application requires additional arguments to work.";
    throw std::runtime_error (sstream.str ());
  }

  std::vector<option> long_options;

  auto createLongOptions = [this](std::vector<option>& long_options)
  {
    iterateArgs ([&long_options](const Arg& arg, const Callback&)
    {
      long_options.push_back ({arg.first.c_str(), required_argument, 0, 0});
    });
    long_options.push_back ({0, 0, 0, 0});
  };

  createLongOptions (long_options);

  int opt = 0;
  int long_index = 0;

  auto callLong = [&long_options, this](int index)
  {
    try
    {
      const auto& option = long_options[index];
      auto callbackIt = m_callbacks.find (std::make_pair(option.name, 0));
      if (callbackIt != m_callbacks.end ())
      {
        std::string oarg = std::string (optarg);
        debug ("Call callback for argument: %s %s", option.name, oarg.c_str());
        callbackIt->second (oarg);
      }
    }
    catch (const std::exception& exception)
    {
      std::stringstream sstream;
      const auto& option = long_options[index];
      sstream << "Exception during parsing \"" << option.name << "\" argument. What: " << exception.what();
      throw std::runtime_error(sstream.str ());
    }
  };

  auto resetOptGet = []()
  {
    optind = 1;
  };

  resetOptGet ();

  while ((opt = getopt_long_only (argc, argv, "", long_options.data(), &long_index)) != -1)
  {
    switch (opt)
    {
      case 0:
        callLong (long_index);
      break;
    }
  }
}
}
