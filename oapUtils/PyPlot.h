#ifndef PYPLOT_H
#define PYPLOT_H

#include "Math.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <memory>
#include "DebugLogs.h"

namespace oap { namespace pyplot {

template<typename T>
std::ostream& operator<< (std::ostream& stream, const std::vector<T>& vec)
{
  stream << "[";
  const size_t sizeM1 = vec.size() - 1;

  for (size_t idx = 0; idx < sizeM1; ++idx)
  {
    stream << vec[idx];
    stream << ", ";
  }

  stream << vec[sizeM1];
  stream << "]";

  return stream;
}

class Feature
{
  public:
    Feature ()
    {}

    Feature (const std::string& formatString) : m_formatString(formatString)
    {}

    Feature (const std::vector<floatt>& values) : m_values(values)
    {}

    Feature (const std::string& formatString, const std::vector<floatt>& values) : m_formatString(formatString), m_values(values)
    {}

    std::string m_formatString;
    std::vector<floatt> m_values;
};

class DataSet
{
  public:
    std::vector<Feature> m_features;
};

using Data = std::map<size_t, DataSet>;

template<typename Coordinates>
void convert(Data& data, const Coordinates& coordinates)
{
  for (const auto& coord : coordinates)
  {
    size_t id = coord.getSetId();
    auto it = data.find (id);
    if (it == data.end ())
    {
      data[id] = DataSet ();
    }

    DataSet& dataSet = data [id];
    dataSet.m_features.resize (coord.size());
    for (size_t idx = 0; idx < coord.size(); ++idx)
    {
      dataSet.m_features[idx].m_values.push_back (coord.at (idx));
      dataSet.m_features[idx].m_formatString = coord.getFormatString (idx);
    }
  }
}

template<typename Coordinates>
Data convert(const Coordinates& coordinates)
{
  Data data;
  convert (data, coordinates);
  return data;
}

void plot2D(const std::string& filePath, const Data& data, const std::vector<size_t>& indecies)
{
  const std::vector<std::pair<std::string, std::string>> imports = {std::make_pair("matplotlib.pyplot","plt")};
  
  auto deleter = [](std::ofstream* stream)
  {
    stream->close();
    delete stream;
  };

  std::unique_ptr<std::ofstream, decltype(deleter)> file (new std::ofstream(filePath, std::ofstream::out), deleter);

  for (const auto& import : imports)
  {
    if (!import.first.empty ())
    {
      *file << "import " << import.first;
    }

    if (!import.second.empty ())
    {
      *file << " as " << import.second;
    }

    *file << std::endl;
  }

  auto printArray = [&file](const std::string& marker, const std::vector<floatt>& values)
  {
    std::stringstream stream;
    stream << marker << " = " << values;
    stream << std::endl;
    *file << stream.str();
  };

  for (const auto& dataSetId : data)
  {
    const auto& dataSet = dataSetId.second;
    size_t id = dataSetId.first;

    auto getIndex = [&indecies] (size_t idx)
    {
      if (indecies.empty())
      {
        return idx;
      }
      return indecies[idx];
    };

    auto getSize = [&indecies, &dataSet]()
    {
      if (indecies.empty())
      {
        return dataSet.m_features.size();
      }
      return indecies.size();
    };

    std::stringstream x_marker;
    x_marker << "x_" << id;
    printArray (x_marker.str(), dataSet.m_features[getIndex(0)].m_values);

    for (size_t idx = 1; idx < getSize(); ++idx)
    {
      std::stringstream y_marker;
      size_t dataIdx = getIndex(idx);

      y_marker << "y_" << id << "_" << idx;

      printArray (y_marker.str (), dataSet.m_features[dataIdx].m_values);

      *file << "plt.plot(" << x_marker.str() << "," << y_marker.str () << ",\"" << dataSet.m_features[dataIdx].m_formatString << "\")" << std::endl;
    }
  }
  *file << "plt.show()" << std::endl;
}

void plot2DAll(const std::string& filePath, const Data& data)
{
  std::vector<size_t> indecies;
  plot2D (filePath, data, indecies);
}

template<typename Values>
void plotLinear (const std::string& filePath, std::initializer_list<Values> valuesVec, std::vector<std::string> formatStrings)
{
  debugAssert (valuesVec.size() > 1);

  DataSet dataSet;
  dataSet.m_features.resize (valuesVec.size() + 1);

  size_t vIdx = 0;
  for (auto it = valuesVec.begin(); it != valuesVec.end(); ++it, ++vIdx)
  {
    const auto& values = *it;
    dataSet.m_features[vIdx].m_values.reserve (values.size());

    for (size_t idx = 0; idx < values.size(); ++idx)
    {
      if (dataSet.m_features[0].m_values.size() <= idx)
      {
        dataSet.m_features[0].m_values.push_back (idx);
      }
      dataSet.m_features[vIdx + 1].m_values.push_back (values[idx]);
    }

    dataSet.m_features[vIdx + 1].m_formatString = formatStrings[vIdx];
  }

  Data data;
  data[0] = (dataSet);
  plot2DAll (filePath, data);
}

void plot2D(const std::string& filePath, const Data& data)
{
  plot2D (filePath, data, {0, 1});
}

}}

#endif
