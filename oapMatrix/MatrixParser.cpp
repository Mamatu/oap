#include "MatrixParser.h"
#include <algorithm>

namespace matrixUtils
{

Parser::Parser() : m_text("")
{}

Parser::Parser(const std::string& text) : m_text(text)
{}

Parser::Parser(const Parser& parser) : m_text(parser.m_text)
{}

Parser::~Parser()
{}

void Parser::setText(const std::string& text)
{ m_text = text; }

bool Parser::getColumns(uintt& columns) const
{
  uintt value = 0;
  if (getValue(value, ID_COLUMNS))
  {
    columns = value;
    return true;
  }
  return false;
}

bool Parser::getRows(uintt& rows) const
{
  uintt value = 0;
  if (getValue(value, ID_ROWS))
  {
    rows = value;
    return true;
  }
  return false;
}

bool Parser::getValue (uintt& value, const std::string& id) const
{
  size_t pos = m_text.find(id, 0);
  if (pos == std::string::npos)
  {
    return false;
  }
  pos += id.length() + 1;
  if (pos >= m_text.length())
  {
    return false;
  }
  size_t pos1 = m_text.find_first_not_of("-0123456789", pos);
  value = atoi(m_text.substr(pos, pos1 - pos).c_str());
  return true;
}

void Parser::satof(floatt& output, const std::string& str) const
{
  if (str.find_first_not_of(".-0123456789e") != std::string::npos)
  {
    throw Parser::ParsingException ("Value contains invalid char", str);
  }
  floatt value = 1;
  int index = 0;
  if (str[0] == '-')
  {
    value = -1;
    index = 1;
  }
  const char* cs = str.c_str();
  output = value * atof(cs + index);
}

inline std::pair<std::string::size_type, std::string::size_type> getBracketsPos (const std::string& text, unsigned int which)
{
  size_t pos = 0, pos1 = 0;
  for (int fa = 0; fa < which; ++fa)
  {
    pos = text.find("[", pos1);
    pos1 = text.find("]", pos);
  }

  return std::make_pair (pos, pos1);
}

bool Parser::hasArray (unsigned int which)
{
  auto pair = getBracketsPos (m_text, which);
  return pair.first != std::string::npos;
}

void Parser::getArrayStr (std::string& array, unsigned int which) const
{
  debugAssert (which > 0);

  auto pair = getBracketsPos (m_text, which);

  if (pair.first == std::string::npos)
  {
    throw Parser::ParsingException ("Cannot find \"[\"", m_text);
  }

  if (pair.second == std::string::npos)
  {
    throw Parser::ParsingException ("Cannot find \"]\"", m_text);
  }

  array = m_text.substr (pair.first + 1, pair.second - pair.first - 1);
}

void Parser::getArray (std::vector<floatt>& array, const std::string& arrayStr) const
{
  size_t pos = 0;
  size_t pos1 = std::string::npos;
  do
  {
    pos1 = arrayStr.find_first_of(",|", pos);
    std::string elementStr = arrayStr.substr(pos, pos1 - pos);
    std::string::iterator it = std::remove_if (elementStr.begin(), elementStr.end(), (int (*)(int))std::isspace);
    elementStr.erase(it, elementStr.end());

    parseElement (array, elementStr);

    pos = pos1 + 1;
  } while (pos1 != std::string::npos);
}

void Parser::parseElement (std::vector<floatt>& array, const std::string& elementStr) const
{
  parseFloatsElement (array, elementStr);
}

bool checkClosedSection (std::string::size_type left, std::string::size_type right)
{
  if ((left == std::string::npos) ^ (right == std::string::npos))
  {
    return false;
  }
  
  if (right <= left && left != std::string::npos && right != std::string::npos)
  {
    return false;
  }

  return true;
}

int getCount (const std::string& elementStr, std::string::size_type lb, std::string::size_type rb)
{
  int count = 1;

  if (lb != std::string::npos)
  {
    std::string partStr = elementStr.substr (lb, rb - lb);
    std::string::size_type posDigit1 = partStr.find_first_of (".-0123456789");
    std::string::size_type posDigit2 = partStr.find_first_not_of (".-0123456789", posDigit1);
    count = atoi (partStr.substr (posDigit1, posDigit2 - posDigit1).c_str());
  }

  return count;
}

using PosSec = std::pair<std::string::size_type, std::string::size_type>;
std::string getValueStr (const std::string& elementStr, std::vector<PosSec>&& sections)
{
  std::string str = elementStr;

  for (auto it = sections.begin(); it != sections.end(); ++it)
  {
    auto pair = *it;

    if (pair.first != std::string::npos)
    {
      std::string::size_type len = pair.second - pair.first + 1;
      if (pair.first + len <= str.length())
      {
        str.erase (pair.first, len);
      }
      else
      {
        throw Parser::ParsingException ("Cannot remove section limited by brackets", elementStr);
      }
    }
  }

  return str;
}

void Parser::parseFloatsElement (std::vector<floatt>& array, const std::string& elementStr) const
{
  std::string::size_type posLB = elementStr.find ("<");
  std::string::size_type posRB = elementStr.find (">");

  if (!checkClosedSection (posLB, posRB))
  {
    throw Parser::ParsingException ("Bad use of \"<\" \">\" brackets", elementStr);
  }

  size_t posLP = elementStr.find ("(");
  size_t posRP = elementStr.find (")");

  if (!checkClosedSection (posLP, posRP))
  {
    throw Parser::ParsingException ("Bad use of \"(\" \")\" parentheses", elementStr);
  }

  int count = getCount (elementStr, posLB, posRB);

  std::string valueStr = getValueStr (elementStr, {std::make_pair (posLB, posRB), std::make_pair (posLP, posRP)});

  if (posLB != std::string::npos && posLB != valueStr.size())
  {
    throw Parser::ParsingException ("Value is splitted by \"<\" \">\" \"(\" or \")\"", elementStr);
  }

  floatt value;

  satof (value, valueStr.c_str());

  for (int fa = 0; fa < count; ++fa)
  {
    array.push_back (value);
  }
}

bool Parser::parseArray (unsigned int which)
{
  std::string arrayStr;
  try
  {
    getArrayStr (arrayStr, which);
  }
  catch (const Parser::ParsingException& pe)
  {
    logError ("%s", pe.what ());
    return false;
  }

  m_array.clear();

  try
  {
    getArray (m_array, arrayStr);
  }
  catch (const Parser::ParsingException& pe)
  {
    logError ("%s", pe.what ());
    return false;
  }
  return true;
}

floatt Parser::getValue(uintt index) const
{
  return m_array.at(index);
}

size_t Parser::getLength() const
{ return m_array.size(); }

const floatt* Parser::getData() const
{ return m_array.data(); }

bool HasArray (const std::string& text, unsigned int which)
{
  Parser parser (text);
  return parser.hasArray (which);
}

std::pair<floatt*, size_t> CreateArray (const std::string& text, unsigned int which)
{
  Parser parser (text);
  try
  {
    parser.parseArray(which);
  }
  catch (const Parser::ParsingException& pe)
  {
    logError ("%s", pe.what());
    return std::make_pair<floatt*, size_t>(nullptr, 0);
  }
  size_t length = parser.getLength();
  floatt* array = new floatt[length];
  memcpy(array, parser.getData(), length * sizeof(floatt));
  return std::make_pair/*<floatt*, size_t>*/(array, length);
}

void Parser::ParsingException::merge (const std::string& _msg, const std::string& _code)
{
  exception += _msg;
  exception += ": \"";
  exception += _code;
  exception += "\"";
}

Parser::ParsingException::ParsingException (const std::string& _msg, const std::string& _code)
{
  merge (_msg, _code);
}

Parser::ParsingException::~ParsingException ()
{}

const char* Parser::ParsingException::what () const throw()
{
  return exception.c_str();
}
}
