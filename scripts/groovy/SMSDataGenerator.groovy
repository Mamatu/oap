/**
 *  SMSDataGenerator - generate matrix whose eigenspair
 *  are defined by used
 */

package groovy

import org.jblas.*
import groovy.Utils
import java.math.RoundingMode
import java.util.Random

utils = new Utils()
absError = 0.0000001f

private def newSMSMatrix(eigenvalues, eigenvectors, newMatrix) {
  def diagMatrix = utils.diag(eigenvalues)
  def sMatrix = newMatrix(eigenvectors)
  def sMatrix_1 = Solve.solve(sMatrix, DoubleMatrix.eye(sMatrix.rows));
  def subMatrix = diagMatrix.mmul(sMatrix_1)
  def outMatrix = sMatrix.mmul(subMatrix)
  return outMatrix
}

private def newSMSMatrix(eigenvalues, eigenvectors) {
  return newSMSMatrix(eigenvalues, eigenvectors, {matrix -> return matrix})
}

def newSMSMatrixRowOrder(eigenvalues, eigenvectors) {
  return newSMSMatrix(eigenvalues, eigenvectors, {evs -> utils.newMatrixRowOrder(evs)})
}

def newSMSMatrixColumnOrder(eigenvalues, eigenvectors) {
  return newSMSMatrix(eigenvalues, eigenvectors, {evs -> new DoubleMatrix(evs as double[][])})
}

assert newSMSMatrixRowOrder([1,1,1],[[1,0,0],[0,1,0],[0,0,1]]).toArray() == [1,0,0,0,1,0,0,0,1]

assert newSMSMatrixColumnOrder([1,-2,2],[[1,0,-1],[1,1,1],[-1,2,-1]]) == newSMSMatrixRowOrder([1,-2,2],[[1,1,-1],[0,1,2],[-1,1,-1]])

def testExpectedArray = [1,-8,-5,-8,4,-8,-5,-8,1]
def mode = new Tuple(3, RoundingMode.HALF_UP)

def newScaledArrayDiv = { array, divValue ->
  array.eachWithIndex { val, idx ->  array[idx] = val.div(divValue) }
  array = utils.newScaledArray(array, mode[0], mode[1])
}

testExpectedArray = newScaledArrayDiv(testExpectedArray, 6)

assert utils.newScaledArray(newSMSMatrixRowOrder([1,-2,2],[[1,0,-1],[1,1,1],[-1,2,-1]]).toArray(), mode[0], mode[1]) == testExpectedArray

assert utils.newScaledArray(newSMSMatrixColumnOrder([1,-2,2],[[1,1,-1],[0,1,2],[-1,1,-1]]).toArray(), mode[0], mode[1]) == testExpectedArray

def createSMSDataHeader(headername, eigenvalues, eigenvectors, smsMatrix) {
  def convertToCArray = { matrix ->
    def arrayStr = []
    def msize = matrix.columns * matrix.rows
    utils.iterate(matrix) { c, r, value ->
      def extra = ""
      if (arrayStr.size() < msize - 1) { extra += ","}
      if (arrayStr.size() != 0 && (arrayStr.size() + 1) % 10 == 0) { extra += "\n" }
      arrayStr.add("${value}${extra}")
    }
    return "{${arrayStr.join(" ")}}"
  }

  if (eigenvalues.rows == 1 && eigenvalues.columns != 1) {
    throw new Exception("No supported case rows == 1 and columns != 1")
  }

  def datastr = 
  """
    #ifndef ${headername.toUpperCase()}_H
    #define ${headername.toUpperCase()}_H
  """.stripIndent().trim() + "\n\n"

  datastr += "namespace ${headername} {\n\n"

  datastr += "double smsmatrix[] =\n${convertToCArray(smsMatrix)};\n\n"

  datastr += "double eigenvalues[] =\n${convertToCArray(eigenvalues)};\n\n"

  datastr += "double eigenvectors[] =\n${convertToCArray(eigenvectors)};\n\n"

  datastr += "}\n\n"
  datastr += "#endif"

  return datastr
}

def createSMSDataTest(testname, eigenvalues, eigenvectors, smsMatrix) {
  def oaptestname = "Oap${testname}"

  def createTests = { matrix, name ->
    def datastr = "TEST_F(${oaptestname}, Load_${name}_Test) {\n"
    datastr += "  uintt columns = ${matrix.columns};\n"
    datastr += "  uintt rows = ${matrix.rows};\n"
    datastr += "  oap::HostMatrixPtr ${name} = host::NewMatrixCopy<double>(columns, rows, (double*)SmsData1::${name}, NULL);\n"
    def builder = new StringBuilder()
    utils.iterate(matrix) { c, r, value ->
      builder.append("  EXPECT_NEAR(${name}->reValues[GetIndex(${name}, ${c}, ${r})], ${value}, ${absError});\n")
    }
    datastr += builder.toString()
    datastr += "}"
    return datastr
  }

  def datastr = """
    #include <gmock/gmock.h>
    #include <gtest/gtest.h>

    #include "HostMatrixUtils.h"
    #include "oapHostMatrixPtr.h"
    #include "MatrixAPI.h"

    #include "${testname}.h"

    class ${oaptestname} : public testing::Test {
      public:
        virtual void SetUp() {}

        virtual void TearDown() {}
    };\n
  """.stripIndent()
  datastr += createTests(smsMatrix, "smsmatrix")
  datastr += "\n\n"
  datastr += createTests(eigenvectors, "eigenvectors")
  datastr += "\n\n"
  datastr += createTests(eigenvalues, "eigenvalues")

  return datastr.toString()
}

def createSMSDataBinary(eigenvalues, eigenvectors, smsMatrix) {
  def array = []
  array[0] = utils.getBinaryData(smsMatrix)
  array[1] = utils.getBinaryData(eigenvalues)

  eigenvectors.eachWithIndex { vec, idx ->
    array[idx + 2] = utils.getBinaryData(vec)
  }

  return array
}

private def createSMSDataHeaderFile(testname, eigenvalues, eigenvectors, smsMatrix, dir = "header") {
  def data = createSMSDataHeader(testname, eigenvalues, eigenvectors, smsMatrix)
  return ["${dir}/${testname}.h" : data]
}

private def createSMSDataLoadTestFile(testname, eigenvalues, eigenvectors, smsMatrix, dir = "loadTest") {
  def headerdata = createSMSDataHeader(testname, eigenvalues, eigenvectors, smsMatrix)
  def data = createSMSDataTest(testname, eigenvalues, eigenvectors, smsMatrix)

  def files = ["${dir}/${testname}.h" : headerdata, "${dir}/${testname}.cpp" : data]

  return files
}

private def createSMSDataLoadTestBinaryFile(testname, eigenvalues, eigenvectors, smsMatrix, dir = "loadTestBinary") {
  def binaryFiles = createSMSDataBinaryFile(testname, eigenvalues, eigenvectors, smsMatrix, dir)
  def testFiles = createSMSDataLoadTestFile(testname, eigenvalues, eigenvectors, smsMatrix, dir)

  binaryFiles.putAll(testFiles)

  return binaryFiles
}

private def createSMSDataBinaryFile(testname, eigenvalues, eigenvectors, smsMatrix, dir = "binary") {
  def data = createSMSDataBinary(eigenvalues, eigenvectors, smsMatrix)
  def files = [:]
  files["${dir}/${testname}/smsmatrix.matrix"] = data[0]
  files["${dir}/${testname}/eigenvalues.matrix"] = data[1]
  for (def idx = 2; idx < data.size(); ++idx) {
    files["${dir}/${testname}/eigenvector${idx - 1}.matrix"] = data[idx]
  }
  return files
}

modeClosures =
    ["binary" : this.&createSMSDataBinaryFile,
     "header" : this.&createSMSDataHeaderFile,
     "load_header_test" : this.&createSMSDataLoadTestFile,
     "load_header_binary_test" : this.&createSMSDataLoadTestBinaryFile]

def allList = new ArrayList<String>()
allList.addAll(modeClosures.values()) 
modeClosures.put("all", allList)

def createRandomSMSData(testname, matrixSize, createSMSDataClosure) {
  def random = new Random()
  def eigenvalues = []
  def eigenvectors = []
  matrixSize.times { idx ->
    eigenvalues[idx] = random.nextDouble()
    eigenvectors[idx] = []
    matrixSize.times { idx1 ->
      eigenvectors[idx][idx1] = random.nextDouble()
    }
  }

  eigenvalues = utils.newMatrixColumnOrder(eigenvalues)
  eigenvectors = utils.newMatrixColumnOrder(eigenvectors as double[][])

  def smsMatrix = newSMSMatrix(eigenvalues, eigenvectors)

  return createSMSDataClosure(testname, eigenvalues, eigenvectors, smsMatrix)
}

def generateData(List<Object> objects) {
  if (objects.size() < 3) {
    throw new IllegalArgumentException("generateData needs more than 2 args.")
  }
  def testsCount = Integer.valueOf(objects[0])
  def matrixSize = Integer.valueOf(objects[1])
  def modes = new ArrayList<String>()
  for (def idx = 2; idx < objects.size(); ++idx) {
    modes.add(objects[idx])
  }
  generateData(testsCount, matrixSize, modes)
}

def generateData(testsCount, matrixSize, List<String> modes) {
  testsCount.times { idx ->

    def closures = new ArrayList<Closure>();
    for (String mode : modes) {
      def createSMSDataClosureMode = modeClosures["${mode}"]
      if (createSMSDataClosureMode == null) {
        throw new Exception("Invalid mode ${mode}. Should be ${modeClosures.keySet()}.")
      }
      if (createSMSDataClosureMode instanceof List) {
        closures.addAll(createSMSDataClosureMode)
      } else {
        closures.add(createSMSDataClosureMode)
      }
    }

    final def dir = "/tmp/Oap/smsdata/"
    def dataFiles = createRandomSMSData("SmsData${idx + 1}", matrixSize, {
      testname, eigenvalues, eigenvectors, smsMatrix ->
      def dataFiles = [:]
      for (def closure : closures) {
        println closure
        dataFiles.putAll(closure(testname, eigenvalues, eigenvectors, smsMatrix))
      }
      return dataFiles
    })

    dataFiles.each { filename, data ->

      def makeSubDirs = {
        def array = filename.split("/")
        if (array.size() > 1) {
          def subdir = array[0..array.size()-2].join("/")
          new File("${dir}/${subdir}").mkdirs()
        }
      }.call()

      utils.saveToFile(dir + filename, data)
    }
  }
}

try {
  if (args) {
    "${args.head()}"( args.tail() as List<Object>)
  }
} catch (Exception e) {
  e.printStackTrace()
}
