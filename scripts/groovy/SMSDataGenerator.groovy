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
  def datastr = 
  """
    #ifndef ${headername.toUpperCase()}_H
    #define ${headername.toUpperCase()}_H
  """.stripIndent().trim() + "\n\n"

  datastr += "namespace ${headername} {\n\n"
  datastr += "const char* smsMatrix = \"${smsMatrix}\";\n\n"

  datastr += "const char* eigenvalues = \"${eigenvalues}\";\n\n"
  for (def idx = 0; idx < eigenvalues.size(); ++idx) {
    def vector = eigenvectors[idx]
    datastr += "const char* eigenvector${idx} = \"${vector}\";\n\n"
  }

  datastr += "}\n\n"
  datastr += "#endif"

  return datastr
}

def createSMSDataBinary(eigenvalues, eigenvectors, smsMatrix) {
  return [Integer.toBinaryString(1),Integer.toBinaryString(1),Integer.toBinaryString(1),Integer.toBinaryString(1)] 
}

private def createSMSDataHeaderFile(testname, eigenvalues, eigenvectors, smsMatrix) {
  def data = createSMSDataHeader(testname, eigenvalues, eigenvectors, smsMatrix)
  return ["${testname}.h" : data]
}

private def createSMSDataBinaryFile(testname, eigenvalues, eigenvectors, smsMatrix) {
  def data = createSMSDataBinary(eigenvalues, eigenvectors, smsMatrix)
  def files = [:]
  files["${testname}/smsmatrix.matrix"] = data[0]
  files["${testname}/eigenvalues.matrix"] = data[1]
  for (def idx = 2; idx < data.size(); ++idx) {
    files["${testname}/eigenvector${idx - 1}.matrix"] = data[idx]
  }
  return files
}

modeClosures =
    ["binary" : this.&createSMSDataBinaryFile,
     "header" : this.&createSMSDataHeaderFile]

def createSMSData(testname, matrixSize, newSMSMatrixClosure, createSMSDataClosure) {
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

def generateData(testsCount, matrixSize, mode) {
  testsCount.times { idx ->

    def createSMSDataClosureObj = modeClosures["${mode}"]
    if (createSMSDataClosureObj == null) {
      throw new Exception("Invalid mode ${mode}. Should be ${modeClosures.keySet()}.")
    }

    final def dir = "/tmp/Oap/smsdata/"
    def dataFiles = createSMSData("smsdata${idx + 1}", matrixSize, this.&newSMSMatrix, createSMSDataClosureObj)
    
    dataFiles.each { filename, data ->

      def makeSubDirs = {
        def array = filename.split("/")
        if (array.size() > 1) {
          def subdir = array[0..array.size()-2].join("/")
          new File("${dir}/${subdir}").mkdirs()
        }
      }.call()

      new File(dir + filename).write(data)
    }
  }
}

try {
  if (args) {
    iargs = []
    args.tail().eachWithIndex { val, idx ->
      try {
        iargs[idx] = Integer.valueOf(val)
      } catch (Exception e) {
        iargs[idx] = val
      }
    }
    "${args.head()}"( iargs )
  }
} catch (Exception e) {
  e.printStackTrace()
}
