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

private def newSMSMatrix(eigenvalues, eigenvectors, newMatrix) {
  def diagMatrix = utils.diag(eigenvalues)
  def sMatrix = newMatrix(eigenvectors)
  def sMatrix_1 = Solve.solve(sMatrix, DoubleMatrix.eye(sMatrix.rows));
  def subMatrix = diagMatrix.mmul(sMatrix_1)
  def outMatrix = sMatrix.mmul(subMatrix)
  return outMatrix
}

def newSMSMatrixRowOrder(eigenvalues, eigenvectors) {
  return newSMSMatrix(eigenvalues, eigenvectors, {evs -> utils.newMatrixRowOrder(evs)})
}

def newSMSMatrix(eigenvalues, eigenvectors) {
  return newSMSMatrix(eigenvalues, eigenvectors, {evs -> new DoubleMatrix(evs as double[][])})
}

assert newSMSMatrixRowOrder([1,1,1],[[1,0,0],[0,1,0],[0,0,1]]).toArray() == [1,0,0,0,1,0,0,0,1]

assert newSMSMatrix([1,-2,2],[[1,0,-1],[1,1,1],[-1,2,-1]]) == newSMSMatrixRowOrder([1,-2,2],[[1,1,-1],[0,1,2],[-1,1,-1]])

def testExpectedArray = [1,-8,-5,-8,4,-8,-5,-8,1]
def mode = new Tuple(3, RoundingMode.HALF_UP)

def newScaledArrayDiv = { array, divValue ->
  array.eachWithIndex { val, idx ->  array[idx] = val.div(divValue) }
  array = utils.newScaledArray(array, mode[0], mode[1])
}

testExpectedArray = newScaledArrayDiv(testExpectedArray, 6)

assert utils.newScaledArray(newSMSMatrixRowOrder([1,-2,2],[[1,0,-1],[1,1,1],[-1,2,-1]]).toArray(), mode[0], mode[1]) == testExpectedArray

assert utils.newScaledArray(newSMSMatrix([1,-2,2],[[1,1,-1],[0,1,2],[-1,1,-1]]).toArray(), mode[0], mode[1]) == testExpectedArray

def createSMSDataHeader(headername, eigenvalues, eigenvectors, smsMatrix) {
  def convertToCArray = { array ->
    if (array instanceof DoubleMatrix) {
      array = array.toArray()
    }
    def arrayStr = []
    array.each {  value ->
      def valueStr = String.format("%.14f", value) ;
      def extra = ""
      def pre = ""
      if (value > 0) { pre = " " }
      if (arrayStr.size() < array.size() - 1) { extra += ","}
      if (arrayStr.size() != 0 && (arrayStr.size() + 1) % 10 == 0) { extra += "\n" }
      arrayStr.add("${pre}${valueStr}${extra}")
    }
    return "{${arrayStr.join(" ")}}"
  }

  def datastr = 
  """
    #ifndef ${headername.toUpperCase()}_H
    #define ${headername.toUpperCase()}_H
  """.stripIndent().trim() + "\n\n"

  datastr += "namespace ${headername} {\n\n"
  datastr += "double smsMatrix[] =\n${convertToCArray(smsMatrix)};\n\n"

  datastr += "double eigenvalues[] =\n${convertToCArray(eigenvalues)};\n\n"

  def elist = []
  for (def idx = 0; idx < eigenvalues.size(); ++idx) {
    def vector = eigenvectors[idx]
    elist.add("${convertToCArray(vector)}")
  }

  datastr += "double eigenvectors[${eigenvalues.size()}][${smsMatrix.rows}] = {\n${elist.join(',\n')}\n};\n"
  
  datastr += "}\n\n"
  datastr += "#endif"

  return datastr
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

private def createSMSDataHeaderFile(testname, eigenvalues, eigenvectors, smsMatrix) {
  def data = createSMSDataHeader(testname, eigenvalues, eigenvectors, smsMatrix)
  return ["header/${testname}.h" : data]
}

private def createSMSDataBinaryFile(testname, eigenvalues, eigenvectors, smsMatrix) {
  def data = createSMSDataBinary(eigenvalues, eigenvectors, smsMatrix)
  def files = [:]
  files["binary/${testname}/smsmatrix.matrix"] = data[0]
  files["binary/${testname}/eigenvalues.matrix"] = data[1]
  for (def idx = 2; idx < data.size(); ++idx) {
    files["binary/${testname}/eigenvector${idx - 1}.matrix"] = data[idx]
  }
  return files
}

modeClosures =
    ["binary" : this.&createSMSDataBinaryFile,
     "header" : this.&createSMSDataHeaderFile]

def createRandomSMSData(testname, matrixSize, newSMSMatrixClosure, createSMSDataClosure) {
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

  def smsMatrix = newSMSMatrixClosure(eigenvalues, eigenvectors)

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
      closures.add(createSMSDataClosureMode)
    }

    final def dir = "/tmp/Oap/smsdata/"
    def dataFiles = createRandomSMSData("smsdata${idx + 1}", matrixSize, this.&newSMSMatrix, {
      testname, eigenvalues, eigenvectors, smsMatrix ->
      def dataFiles = [:]
      for (def closure : closures) {
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
