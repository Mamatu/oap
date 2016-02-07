#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H

#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include "Matrix.h"
#include "MatrixAPI.h"

namespace matrixUtils {

extern const char* ID_COLUMNS;
extern const char* ID_ROWS;
extern const char* ID_LENGTH;

template <typename T>
class OccurencesList : public std::vector<std::pair<uintt, T> > {};

template <typename T>
class SubArrays : public std::vector<std::pair<T*, uintt> > {};

class Range {
 protected:
  uintt m_bcolumn;
  uintt m_columns;
  uintt m_brow;
  uintt m_rows;

 public:
  Range(uintt bcolumn, uintt columns, uintt brow, uintt rows);

  virtual ~Range();

  virtual uintt getBColumn() const;
  virtual uintt getEColumn() const;
  virtual uintt getColumns() const;

  virtual uintt getBRow() const;
  virtual uintt getERow() const;
  virtual uintt getRows() const;
};

class MatrixRange : public Range {
 protected:
  const math::Matrix* m_matrix;

  void getSubArrays(SubArrays<floatt>& subArrays, floatt* array,
                    const math::Matrix* matrix) const;

 public:
  MatrixRange(const math::Matrix* matrix);
  MatrixRange(const math::Matrix* matrix, const Range& range);
  MatrixRange(const math::Matrix* matrix, uintt bcolumn, uintt columns,
              uintt brow, uintt rows);

  virtual ~MatrixRange();

  bool isReValues() const;
  bool isImValues() const;

  virtual uintt getEColumn() const;
  virtual uintt getERow() const;

  const math::Matrix* getMatrix() const;

  void getReSubArrays(SubArrays<floatt>& subArrays) const;
  void getImSubArrays(SubArrays<floatt>& subArrays) const;
};

template <typename T>
void PrepareOccurencesList(OccurencesList<T>& occurencesList, T* array,
                           uintt length, bool repeats, T zeroLimit) {
  for (uintt fa = 0; fa < length; ++fa) {
    floatt value = array[fa];
    if (-zeroLimit < value && value < zeroLimit) {
      value = 0.f;
    }
    if (repeats == false || occurencesList.size() == 0 ||
        occurencesList[occurencesList.size() - 1].second != value) {
      occurencesList.push_back(std::make_pair<uintt, floatt>(1, value));
    } else {
      occurencesList[occurencesList.size() - 1].first++;
    }
  }
}

template <typename T>
void PrintArrays(std::string& output, T** arrays, uintt* lengths, uintt count,
                 T zeroLimit, bool repeats = true, bool pipe = true,
                 bool endl = true,
                 size_t sectionLength = std::numeric_limits<size_t>::max()) {
  output = "[";
  OccurencesList<T> valuesVec;
  uintt totalLength = 0;
  for (uintt index = 0; index < count; ++index) {
    T* array = arrays[index];
    uintt length = lengths[index];
    totalLength += length;
    PrepareOccurencesList(valuesVec, array, length, repeats, zeroLimit);
  }

  std::stringstream sstream;
  for (uintt fa = 0; fa < valuesVec.size(); ++fa) {
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
  sstream << "] (" << ID_LENGTH << "=" << totalLength;
  output += sstream.str();
  output += ")\n";
}

template <typename T>
void PrintArrays(std::string& output, const SubArrays<T>& subArrays,
                 T zeroLimit, bool repeats = true, bool pipe = true,
                 bool endl = true,
                 size_t sectionLength = std::numeric_limits<size_t>::max()) {
  uintt count = subArrays.size();
  T** arrays = new T* [count];
  uintt* lengths = new uintt[count];
  for (uintt fa = 0; fa < count; ++fa) {
    arrays[fa] = subArrays[fa].first;
    lengths[fa] = subArrays[fa].second;
  }
  PrintArrays(output, arrays, lengths, count, zeroLimit, repeats, pipe, endl,
              sectionLength);
  delete[] arrays;
  delete[] lengths;
}

template <typename T>
void PrintArray(std::string& output, T* array, uintt length, T zeroLimit,
                bool repeats = true, bool pipe = true, bool endl = true,
                size_t sectionLength = std::numeric_limits<size_t>::max()) {
  T* arrays[] = {array};
  uintt lengths[] = {length};
  PrintArrays(output, arrays, lengths, 1, zeroLimit, repeats, pipe, endl,
              sectionLength);
}

template <typename T>
void PrintReValues(std::string& output, const MatrixRange& matrixRange,
                   T zeroLimit, bool repeats = true, bool pipe = true,
                   bool endl = true,
                   size_t sectionLength = std::numeric_limits<size_t>::max()) {
  SubArrays<floatt> subArrrays;
  matrixRange.getReSubArrays(subArrrays);
  PrintArrays(output, subArrrays, zeroLimit, repeats, pipe, endl,
              sectionLength);
}

template <typename T>
void PrintImValues(std::string& output, const MatrixRange& matrixRange,
                   T zeroLimit, bool repeats = true, bool pipe = true,
                   bool endl = true,
                   size_t sectionLength = std::numeric_limits<size_t>::max()) {
  SubArrays<floatt> subArrrays;
  matrixRange.getImSubArrays(subArrrays);
  PrintArrays(output, subArrrays, zeroLimit, repeats, pipe, endl,
              sectionLength);
}

inline void PrintMatrix(std::string& output, const MatrixRange& matrixRange,
                        floatt zeroLimit = 0, bool repeats = true,
                        bool pipe = true, bool endl = true) {
  std::stringstream sstream;
  sstream << "(" << ID_COLUMNS << "=" << matrixRange.getColumns() << ", "
          << ID_ROWS << "=" << matrixRange.getRows() << ") ";
  output += sstream.str();
  std::string output1;
  if (matrixRange.isReValues()) {
    PrintReValues(output1, matrixRange, zeroLimit, repeats, pipe, endl,
                  matrixRange.getColumns());
    output += output1 + " ";
  }
  if (matrixRange.isImValues()) {
    PrintImValues(output1, matrixRange, zeroLimit, repeats, pipe, endl,
                  matrixRange.getColumns());
    output += output1;
  }
}

inline void PrintMatrix(std::string& output, const math::Matrix* matrix,
                        floatt zeroLimit = 0, bool repeats = true,
                        bool pipe = true, bool endl = true) {
  if (matrix == NULL) {
    return;
  }
  MatrixRange matrixRange(matrix);
  PrintMatrix(output, matrixRange, zeroLimit, repeats, pipe, endl);
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
