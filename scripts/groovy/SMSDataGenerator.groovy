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

def createSMSData(filepath, eigenvalues, eigenvectors, newSMSMatrix) {
  def smsMatrix = newSMSMatrix(eigenvalues, eigenvectors)
  
  def datastr = "const char* smsMatrix = \"[${smsMatrix}]\";\n"

  for (def idx = 0; idx < eigenvalues.size(); ++idx) {
    def value = eigenvalues[idx]
    def vector = eigenvectors[idx]
    datastr += "const char* eigenvalue${idx} = \"[${value}]\";\n"
    datastr += "const char* eigenvector${idx} = \"[${vector}]\";\n"
  }

  new File(filepath).write(datastr)
}

def createSMSData(filepath, size) {
  def random = new Random()
  def eigenvalues = []
  def eigenvectors = []
  size.times { idx ->
    eigenvalues[idx] = random.nextDouble()
    eigenvectors[idx] = []
    size.times { idx1 ->
      eigenvectors[idx][idx1] = random.nextDouble()
    }
  }
  return createSMSData(filepath, eigenvalues, eigenvectors, {evalues, evectors -> newSMSMatrix(evalues, evectors)})
}


def testData1() {
  createSMSData("/tmp/data1.h", 10)
}

testData1()
