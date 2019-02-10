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

#ifndef OAP_ARGSPARSER_H
#define OAP_ARGSPARSER_H

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <unistd.h>
#include <getopt.h>

#include <DebugLogs.h>

namespace oap
{
  class IArgsParser
  {
    public:
      IArgsParser () {}
      virtual ~IArgsParser () {}

      virtual void parse (int argc, char **argv) const = 0;
  };

  class ArgsParser : public IArgsParser
  {
    public:
      ArgsParser();
      virtual ~ArgsParser()
      {}

      ArgsParser(const ArgsParser&) = delete;
      ArgsParser(ArgsParser&&) = delete;
      ArgsParser& operator=(const ArgsParser&) = delete;
      ArgsParser& operator=(ArgsParser&&) = delete;

      using Arg = std::pair<std::string, char>;
      using Callback = std::function<void(const std::string& arg)>;

      bool isInitialized (const Arg& arg) const;

      template<typename IterCallback>
      void iterateArgs_while (IterCallback&& iterCallback) const
      {
        for (auto it = m_callbacks.cbegin(); it != m_callbacks.cend(); ++it)
        {
          bool bcontinue = iterCallback (it->first, it->second);
          if (!bcontinue)
          {
            return;
          }
        }
      }

      template<typename IterCallback>
      void iterateArgs (IterCallback&& iterCallback) const
      {
        iterateArgs_while ([&iterCallback](const Arg& arg, const Callback& callback)
        {
          iterCallback (arg, callback);
          return true; 
        });
      }

      Arg getArg (const std::string& larg) const;

      Arg getArg (char sarg) const;

      bool containsArg (const std::string& larg) const;

      bool containsArg (char sarg) const;

      void registerArg (const std::string& larg, const Callback& callback);

      void parse (int argc, char **argv) const override;

    private:
      char m_sarg = 0;
      std::map<Arg, Callback> m_callbacks;

      template<typename T>
      void throwsIfContains (T arg)
      {
        if (containsArg (arg))
        {
          std::stringstream sstream;
          sstream << "Args contains registered \"" << arg << "\" argument.";
          throw std::runtime_error (sstream.str ());
        }
      }
  };
}

#endif

