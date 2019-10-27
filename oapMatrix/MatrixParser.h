/*
 * Copyright 2016 - 2019 Marcin Matula
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

#ifndef OAP_MATRIX_PARSER_H
#define OAP_MATRIX_PARSER_H

#include "MatrixUtils.h"

namespace matrixUtils
{

class Parser
{
 private:
  std::string m_text;
  std::vector<floatt> m_array;

 protected:
  bool getValue(uintt& value, const std::string& id) const;
  void getArrayStr(std::string& array, unsigned int which) const;
  void getArray(std::vector<floatt>& array, const std::string& arrayStr) const;

  void parseElement(std::vector<floatt>& array,
                    const std::string& elementStr) const;

  void isOneElement(const std::string& elementStr) const;
  void parseFloatElement(std::vector<floatt>& array,
                         const std::string& elementStr) const;
  void parseFloatsElement(std::vector<floatt>& array,
                          const std::string& elementStr) const;

  void satof(floatt& output, const std::string& str) const;

 public:
  Parser();
  Parser(const std::string& text);
  Parser(const Parser& parser);
  virtual ~Parser();

  Parser(Parser&& parser) = delete;
  Parser& operator=(const Parser&) = delete;
  Parser& operator=(Parser&&) = delete;

  bool hasArray (unsigned int which);
  void setText(const std::string& text);

  bool getColumns(uintt& columns) const;
  bool getRows(uintt& rows) const;

  void parseArray(unsigned int which);

  floatt getValue(uintt index) const;

  size_t getLength() const;
  const floatt* getData() const;

  class ParsingException : public std::exception
  {
      std::string exception;

      void merge (const std::string& _msg, const std::string& _code);

    public:
      ParsingException (const std::string& _msg, const std::string& _code);

      virtual ~ParsingException ();

      virtual const char* what () const throw() override;
  };
};

  bool HasArray (const std::string& text, unsigned int which);
  std::pair<floatt*, size_t> CreateArray (const std::string& text, unsigned int which);
}

#endif

