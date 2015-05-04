#include "MatrixUtils.h"
#include <algorithm>
#include <locale>
#include <math.h>
#include <stdlib.h>
#include <DebugLogs.h>

namespace matrixUtils {

const char* ID_COLUMNS = "columns";
const char* ID_ROWS = "rows";
const char* ID_LENGTH = "length";

Parser::Parser() : m_text("") {}
Parser::Parser(const std::string& text) : m_text(text) {}
Parser::Parser(const Parser& parser) : m_text(parser.m_text) {}
Parser::~Parser() {}

void Parser::setText(const std::string& text) { m_text = text; }

bool Parser::getColumns(uintt& columns) const {
  uintt value = 0;
  bool status = getValue(value, ID_COLUMNS);
  columns = value;
  return status;
}

bool Parser::getRows(uintt& rows) const {
  uintt value = 0;
  bool status = getValue(value, ID_ROWS);
  rows = value;
  return status;
}

bool Parser::getValue(uintt& value, const std::string& id) const {
  size_t pos = m_text.find(id, 0);
  if (pos == std::string::npos) {
    return false;
  }
  pos += id.length() + 1;
  if (pos >= m_text.length()) {
    return false;
  }
  size_t pos1 = m_text.find_first_not_of("-0123456789", pos);
  value = atoi(m_text.substr(pos, pos1 - pos).c_str());
  return true;
}

floatt Parser::satof(const std::string& str) const {
  floatt value = 1;
  int index = 0;
  if (str[0] == '-') {
    value = -1;
    index = 1;
  }
  const char* cs = str.c_str();
  return value * atof(cs + index);
}

bool Parser::getArrayStr(std::string& array, unsigned int which) const {
  debugAssert(which > 0);
  size_t pos = 0;
  pos = m_text.find("[", pos);
  for (int fa = 0; fa < which - 1; ++fa) {
    pos = m_text.find("[", pos + 1);
  }
  if (pos == std::string::npos) {
    return false;
  }
  size_t pos1 = m_text.find("]", pos);
  if (pos1 == std::string::npos) {
    return false;
  }
  ++pos;
  array = m_text.substr(pos, pos1 - pos);
}

void Parser::getArray(std::vector<floatt>& array,
                      const std::string& arrayStr) const {
  size_t pos = 0;
  size_t pos1 = std::string::npos;
  do {
    pos1 = arrayStr.find(",", pos);
    std::string elementStr = arrayStr.substr(pos, pos1 - pos);
    std::string::iterator it = std::remove_if(
        elementStr.begin(), elementStr.end(), (int (*)(int))std::isspace);
    elementStr.erase(it, elementStr.end());
    parseElement(array, elementStr);
    pos = pos1 + 1;
  } while (pos1 != std::string::npos);
}

void Parser::parseElement(std::vector<floatt>& array,
                          const std::string& elementStr) const {
  if (isOneElement(elementStr)) {
    parseFloatElement(array, elementStr);
  } else {
    parseFloatsElement(array, elementStr);
  }
}

bool Parser::isOneElement(const std::string& elementStr) const {
  return elementStr.find("<") == std::string::npos &&
         elementStr.find(">") == std::string::npos;
}

void Parser::parseFloatElement(std::vector<floatt>& array,
                               const std::string& elementStr) const {
  const floatt v = satof(elementStr.c_str());
  array.push_back(v);
}

void Parser::parseFloatsElement(std::vector<floatt>& array,
                                const std::string& elementStr) const {
  size_t pos = elementStr.find("<");
  size_t pos1 = elementStr.find(">");
  std::string partStr = elementStr.substr(pos, pos1 - pos);
  size_t posDigit1 = partStr.find_first_of("-0123456789");
  size_t posDigit2 = partStr.find_first_not_of("-0123456789", posDigit1);
  int count = atoi(partStr.substr(posDigit1, posDigit2 - posDigit1).c_str());
  std::string sub = elementStr.substr(0, pos);
  const floatt value = satof(sub.c_str());
  for (int fa = 0; fa < count; ++fa) {
    array.push_back(value);
  }
}

bool Parser::parseArray(unsigned int which) {
  std::string arrayStr;
  if (getArrayStr(arrayStr, which) == false) {
    return false;
  }
  m_array.clear();
  getArray(m_array, arrayStr);
  return true;
}

floatt Parser::getValue(uintt index) const { return m_array.at(index); }

size_t Parser::getLength() const { return m_array.size(); }

const floatt* Parser::getData() const { return m_array.data(); }

floatt* CreateArray(const std::string& text, unsigned int which) {
  Parser parser(text);
  if (parser.parseArray(which) == false) {
    return NULL;
  }
  size_t length = parser.getLength();
  floatt* array = new floatt[length];
  memcpy(array, parser.getData(), length * sizeof(floatt));
  return array;
}
}
