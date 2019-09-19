#include <fstream>
#include <sstream>
#include <functional>

#include "PyPlot.h"

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

size_t checkSizes (const DataSet& dataSet)
{
  auto it = dataSet.m_features.begin();

  if (it == dataSet.m_features.end ())
  {
    throw std::runtime_error ("Empty data set");
  }

  size_t size = it->m_values.size();

  for (; it != dataSet.m_features.end(); ++it)
  {
    if (size != it->m_values.size())
    {
      throw std::runtime_error ("Not equal length of feature's values");
    }
  }

  return size;
}

template<typename T>
std::ostream& operator<< (std::ostream& stream, const DataSet& dataSet)
{
  size_t iterSize = checkSizes (dataSet);
  for (size_t dsIdx = 0; dsIdx < iterSize; ++dsIdx)
  {
    for (size_t idx = 0; idx < dataSet.m_features.size() - 1; ++idx)
    {
      stream << dataSet.m_features[idx].m_values[dsIdx] << ",";
    }
    stream << dataSet.m_features[dataSet.m_features.size() - 1].m_values[dsIdx] << std::endl;
  }

  return stream;
}

using Imports = std::vector<std::pair<std::string, std::string>>;

void FileDeleter (std::ofstream* stream)
{
  stream->close();
  delete stream;
}

using FilePtr = std::shared_ptr<std::ofstream>;

void plot2D_PyFile(const std::string& filePath, const Data& data, const std::vector<size_t>& indecies);
void plot2D_PyCsvFile(const std::string& filePath, const Data& data, const std::vector<size_t>& indecies);

void writeImportsHeader (FilePtr filePtr, const Imports& imports)
{
  for (const auto& import : imports)
  {
    if (!import.first.empty ())
    {
      *filePtr << "import " << import.first;
    }

    if (!import.second.empty ())
    {
      *filePtr << " as " << import.second;
    }

    *filePtr << std::endl;
  }
}

void plot2D(const std::string& filePath, const Data& data, const std::vector<size_t>& indecies, FileType fileType)
{
  using Func = std::function<void(const std::string&, const Data&, const std::vector<size_t>&)>;
  const std::map<FileType, Func> funcs =
  {
    {FileType::OAP_PYTHON_FILE, plot2D_PyFile},
    {FileType::OAP_PYTHON_AND_CSV_FILE, plot2D_PyCsvFile}
  };

  funcs.at(fileType)(filePath, data, indecies);
}

std::string getCsvFilePath (const std::string& path)
{
  std::string csvPath;
  std::string::size_type idx = path.find (".py");
  if (idx == std::string::npos)
  {
    csvPath = path;
  }
  else
  {
    csvPath = path.substr (0, idx);
    
  }
  csvPath += "_data.csv";
  return csvPath;
}

void plot2D_PyCsvFile(const std::string& filePath, const Data& data, const std::vector<size_t>& indecies)
{
  debugAssertMsg (false,"not implemented yet");
  const Imports imports =
  {
    std::make_pair("matplotlib.pyplot","plt"),
    std::make_pair("csv",""),
  };

  std::string csvPath = getCsvFilePath (filePath);

  FilePtr pyFile (new std::ofstream(filePath, std::ofstream::out), FileDeleter);
  FilePtr csvFile (new std::ofstream(csvPath, std::ofstream::out), FileDeleter);

  writeImportsHeader (pyFile, imports);
}

void plot2D_PyFile(const std::string& filePath, const Data& data, const std::vector<size_t>& indecies)
{
  const Imports imports =
  {
    std::make_pair("matplotlib.pyplot","plt"),
  };
  
  FilePtr pyFile (new std::ofstream(filePath, std::ofstream::out), FileDeleter);

  writeImportsHeader (pyFile, imports);

  auto printArray = [&pyFile](const std::string& marker, const std::vector<floatt>& values)
  {
    std::stringstream stream;
    stream << marker << " = " << values;
    stream << std::endl;
    *pyFile << stream.str();
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

      *pyFile << "plt.plot(" << x_marker.str() << "," << y_marker.str () << ",\"" << dataSet.m_features[dataIdx].m_formatString << "\")" << std::endl;
    }
  }
  *pyFile << "plt.show()" << std::endl;
}

void plot2DAll(const std::string& filePath, const Data& data, FileType fileType)
{
  std::vector<size_t> indecies;
  plot2D (filePath, data, indecies, fileType);
}

void plot2D(const std::string& filePath, const Data& data, FileType fileType)
{
  plot2D (filePath, data, {0, 1}, fileType);
}

}}
