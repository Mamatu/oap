#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H

#include <vector>
#include <string>
#include <sstream>
#include "Matrix.h"

namespace matrixUtils {

extern const char* ID_COLUMNS;
extern const char* ID_ROWS;
extern const char* ID_LENGTH;

template <typename T>
class OccurencesList : public std::vector<std::pair<uintt, T> > {};

template <typename T>
void PrepareOccurencesList(OccurencesList<T>& occurencesList, T* array,
                           size_t length) {
  for (size_t fa = 0; fa < length; ++fa) {
    floatt value = array[fa];
    if (occurencesList.size() == 0 ||
        occurencesList[occurencesList.size() - 1].second != value) {
      occurencesList.push_back(std::make_pair<uintt, floatt>(1, value));
    } else {
      occurencesList[occurencesList.size() - 1].first++;
    }
  }
}

template <typename T>
void PrintArray(std::string& output, T* array, size_t length) {
  OccurencesList<T> valuesVec;
  PrepareOccurencesList(valuesVec, array, length);
  output = "[";
  std::stringstream sstream;
  for (size_t fa = 0; fa < valuesVec.size(); ++fa) {
    sstream << valuesVec[fa].second;
    if (valuesVec[fa].first > 1) {
      sstream << " <repeats " << valuesVec[fa].first << " times>";
    }
    if (fa < valuesVec.size() - 1) {
      sstream << ", ";
    }
    output += sstream.str();
    sstream.str("");
  }
  sstream.str("");
  sstream << "] (" << ID_LENGTH << "=" << length;
  output += sstream.str();
  output += ")";
}

inline void PrintMatrix(std::string& output, const math::Matrix* matrix) {
  std::stringstream sstream;
  sstream << "(" << ID_COLUMNS << "=" << matrix->columns << ", " << ID_ROWS
          << "=" << matrix->rows << ") ";
  output += sstream.str();
  size_t length = matrix->columns * matrix->rows;
  std::string output1;
  if (matrix->reValues != NULL) {
    PrintArray(output1, matrix->reValues, length);
    output += output1 + " ";
  }
  if (matrix->imValues != NULL) {
    PrintArray(output1, matrix->imValues, length);
    output += output1;
  }
}

class Parser {
 private:
  std::string m_text;
  std::vector<floatt> m_array;

 protected:
  bool getValue(uintt& value, const std::string& id) const;
  bool getArrayStr(std::string& array, unsigned int which) const;
  void getArray(std::vector<floatt>& array, const std::string& arrayStr) const;

  void parseElement(std::vector<floatt>& array,
                    const std::string& elementStr) const;

  bool isOneElement(const std::string& elementStr) const;
  void parseFloatElement(std::vector<floatt>& array,
                         const std::string& elementStr) const;
  void parseFloatsElement(std::vector<floatt>& array,
                          const std::string& elementStr) const;

  floatt satof(const std::string& str) const;

 public:
  Parser();
  Parser(const std::string& text);
  Parser(const Parser& parser);
  virtual ~Parser();

  void setText(const std::string& text);

  bool getColumns(uintt& columns) const;
  bool getRows(uintt& rows) const;
  bool parseArray(unsigned int which);
  floatt getValue(uintt index) const;
  size_t getLength() const;
  const floatt* getData() const;
};

floatt* CreateArray(const std::string& text, unsigned int which);
};

#endif  // MATRIXUTILS_H
