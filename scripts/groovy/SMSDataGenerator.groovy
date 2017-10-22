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

def createSMSData(dir, testname, eigenvalues, eigenvectors, newSMSMatrix) {
  def smsMatrix = newSMSMatrix(eigenvalues, eigenvectors)
  
  def datastr = 
  """
    #ifndef ${testname.toUpperCase()}_H
    #define ${testname.toUpperCase()}_H
  """.stripIndent().trim() + "\n\n"

  datastr += "namespace ${testname} {\n\n"
  datastr += "const char* smsMatrix = \"${smsMatrix}\";\n\n"

  for (def idx = 0; idx < eigenvalues.size(); ++idx) {
    def value = eigenvalues[idx]
    def vector = eigenvectors[idx]
    datastr += "const char* eigenvalue${idx} = \"${value}\";\n"
    datastr += "const char* eigenvector${idx} = \"${vector}\";\n\n"
  }

  datastr += "}\n\n"
  datastr += "#endif"

  new File("${dir}${testname}.h").write(datastr)
}

def createSMSData(dir, testname, matrixSize) {
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
  return createSMSData(dir, testname, eigenvalues, eigenvectors, {evalues, evectors -> newSMSMatrix(evalues, evectors)})
}

def generateData(testsCount, matrixSize) {
  testsCount.times { idx ->
    createSMSData("/tmp/Oap/smsdata/","smsdata${idx + 1}", matrixSize)
  }
}

if (args) {
  iargs = []
  args.tail().eachWithIndex { val, idx ->
    iargs[idx] = Integer.valueOf(val)
  }
  "${args.head()}"( iargs )
}
