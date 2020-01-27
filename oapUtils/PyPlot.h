#ifndef PYPLOT_H
#define PYPLOT_H

#include "Math.h"

#include <string>
#include <vector>
#include <map>

#include <memory>
#include "Logger.h"

namespace oap { namespace pyplot {

enum class FileType
{
  OAP_PYTHON_FILE,
  OAP_PYTHON_AND_CSV_FILE
};

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

using Data = std::map<int, DataSet>;

template<typename Coordinates>
void convert(Data& data, const Coordinates& coordinates)
{
  for (const auto& coord : coordinates)
  {
    int label = coord.getGeneralLabel();
    auto it = data.find (label);
    if (it == data.end ())
    {
      data[label] = DataSet ();
    }

    DataSet& dataSet = data [label];
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

void plot2D(const std::string& filePath, const Data& data, FileType fileType = FileType::OAP_PYTHON_AND_CSV_FILE);
void plot2D(const std::string& filePath, const Data& data, const std::vector<size_t>& indecies, FileType fileType = FileType::OAP_PYTHON_AND_CSV_FILE);

void plot2DAll(const std::string& filePath, const Data& data, FileType fileType = FileType::OAP_PYTHON_AND_CSV_FILE);

template<typename Values>
void plotLinear (const std::string& filePath, std::initializer_list<Values> valuesVec, std::vector<std::string> formatStrings, FileType fileType = FileType::OAP_PYTHON_AND_CSV_FILE)
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
  plot2DAll (filePath, data, fileType);
}

template<typename GetLabel>
void plotCoords2D /*= [&network, &houtput, &hinput]*/ (const std::string& path, const std::tuple<floatt, floatt, floatt>& xRange, const std::tuple<floatt, floatt, floatt>& yRange, GetLabel&& getLabel, const std::vector<std::string>& formatStrings)
{
  class CoordXY
  {
    private:
      floatt x;
      floatt y;
      size_t label;
      const std::vector<std::string>& formatStrings;

    public:
      CoordXY (floatt _x, floatt _y, size_t _label, const std::vector<std::string>& _formatStrings) : x(_x), y(_y), label(_label), formatStrings(_formatStrings)
      {}

      floatt getX() const { return x; }
      floatt getY() const { return y; }
      int getGeneralLabel () const { return static_cast<int>(label); }

      size_t size() const
      {
        return 2;
      }

      floatt at (size_t idx) const
      {
        switch (idx)
        {
          case 0:
          return getX();
          case 1:
          return getY();
        };
        return getY();
      }

      std::string getFormatString (size_t idx) const
      {
        return formatStrings[getGeneralLabel()];
      }
  };

  std::vector<CoordXY> coords;
  //oap::HostMatrixPtr hinput = oap::host::NewReMatrix (1, 3);
  //oap::HostMatrixPtr houtput = oap::host::NewReMatrix (1, 1);

  for (floatt x = std::get<0>(xRange); x < std::get<1>(xRange); x += std::get<2>(xRange))
  {
    for (floatt y = std::get<0>(yRange); y < std::get<1>(yRange); y += std::get<2>(yRange))
    {
     // *GetRePtrIndex (hinput, 0) = x;
     // *GetRePtrIndex (hinput, 1) = y;
     // *GetRePtrIndex (hinput, 2) = 1;

     // network->setInputs (hinput, Network::HOST);
     // network->setExpected (houtput, Network::HOST);

     // network->forwardPropagation ();

     // network->getOutputs (houtput.get(), Network::HOST);
      coords.push_back (CoordXY(x, y, getLabel(x, y), formatStrings));
    }
  }

  oap::pyplot::plot2DAll (path, oap::pyplot::convert (coords), oap::pyplot::FileType::OAP_PYTHON_FILE);
}

}}

#endif
