import org.jblas.*

def diag(array, Class clazz = DoubleMatrix) {
  return clazz.diag(clazz.newInstance(array as double[]))
}

assert diag([1,2,3], DoubleMatrix) == new DoubleMatrix([[1,0,0],[0,2,0],[0,0,3]] as double[][]) 

assert DoubleMatrix.diag(new DoubleMatrix([1,2,3] as double[])) == diag([1,2,3], DoubleMatrix)

assert diag([1,2,3]) == diag([1,2,3], DoubleMatrix)

/**
 * Creates matrix from array. Data in array are sored in row major order [[row1],[row2],[row3]]
 */
def newMatrixRowOrder(array, Class clazz = DoubleMatrix) {

  def columnOrientedArray = []

  array.eachWithIndex { vec, i ->
    vec.eachWithIndex { val, j ->
      if (columnOrientedArray[j] == null) {
        columnOrientedArray[j] = []
      }
      columnOrientedArray[j][i] = val
    }
  }

  return clazz.newInstance(columnOrientedArray as double[][])
}

assert newMatrixRowOrder([[1,2,3],[4,5,6],[7,8,9]]).toArray() == [1,2,3,4,5,6,7,8,9]

assert newMatrixRowOrder([[1,2],[4,5],[7,8]]).toArray() == [1,2,4,5,7,8]


