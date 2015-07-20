#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H

#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include "Matrix.h"

namespace matrixUtils {

extern const char* ID_COLUMNS;
extern const char* ID_ROWS;
extern const char* ID_LENGTH;

template <typename T>
class OccurencesList : public std::vector<std::pair<uintt, T> > {};

template <typename T>
void PrepareOccurencesList(OccurencesList<T>& occurencesList, T* array,
                           size_t length, bool repeats) {
  for (size_t fa = 0; fa < length; ++fa) {
    floatt value = array[fa];
    if (repeats == false || occurencesList.size() == 0 ||
        occurencesList[occurencesList.size() - 1].second != value) {
      occurencesList.push_back(std::make_pair<uintt, floatt>(1, value));
    } else {
      occurencesList[occurencesList.size() - 1].first++;
    }
  }
}

template <typename T>
void PrintArray(std::string& output, T* array, size_t length,
                bool repeats = true, bool pipe = true, bool endl = true,
                size_t sectionLength = std::numeric_limits<size_t>::max()) {
  OccurencesList<T> valuesVec;
  PrepareOccurencesList(valuesVec, array, length, repeats);
  output = "[";
  std::stringstream sstream;
  for (size_t fa = 0; fa < valuesVec.size(); ++fa) {
    sstream << valuesVec[fa].second;
    if (valuesVec[fa].first > 1) {
      sstream << " <repeats " << valuesVec[fa].first << " times>";
    }
    if (fa < valuesVec.size() - 1) {
      bool notEndLine = (fa + 1) % sectionLength != 0;
      if (notEndLine) {
        sstream << ", ";
      } else if (!pipe && !notEndLine) {
        sstream << ", ";
      } else if (pipe && !notEndLine) {
        sstream << " | ";
      }
      if (!notEndLine && endl) {
        sstream << std::endl;
      }
    }
    output += sstream.str();
    sstream.str("");
  }
  sstream.str("");
  sstream << "] (" << ID_LENGTH << "=" << length;
  output += sstream.str();
  output += ")\n";
}

inline void PrintMatrix(std::string& output, const math::Matrix* matrix,
                        bool repeats = true, bool pipe = true,
                        bool endl = true) {
  if (matrix == NULL) {
    return;
  }
  std::stringstream sstream;
  sstream << "(" << ID_COLUMNS << "=" << matrix->columns << ", " << ID_ROWS
          << "=" << matrix->rows << ") ";
  output += sstream.str();
  size_t length = matrix->columns * matrix->rows;
  std::string output1;
  if (matrix->reValues != NULL) {
    PrintArray(output1, matrix->reValues, length, repeats, pipe, endl,
               matrix->columns);
    output += output1 + " ";
  }
  if (matrix->imValues != NULL) {
    PrintArray(output1, matrix->imValues, length, repeats, pipe, endl,
               matrix->columns);
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
  bool getArray(std::vector<floatt>& array, const std::string& arrayStr) const;

  bool parseElement(std::vector<floatt>& array,
                    const std::string& elementStr) const;

  bool isOneElement(const std::string& elementStr) const;
  bool parseFloatElement(std::vector<floatt>& array,
                         const std::string& elementStr) const;
  bool parseFloatsElement(std::vector<floatt>& array,
                          const std::string& elementStr) const;

  bool satof(floatt& output, const std::string& str) const;

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

std::pair<floatt*, size_t> CreateArray(const std::string& text,
                                       unsigned int which);
};

#endif  // MATRIXUTILS_H
