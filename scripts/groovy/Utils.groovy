package groovy

import org.jblas.*
import java.nio.*
import java.nio.file.*

class Utils {

  Utils() {
    UtilsTests()
  }

  private def UtilsTests() {
    diagTests()
    newMatrixRowOrderTests()
  }
  
  def diag(array, Class clazz = DoubleMatrix) {
    return clazz.diag(clazz.newInstance(array as double[]))
  }
  
  private def diagTests() {
    assert diag([1,2,3], DoubleMatrix) == new DoubleMatrix([[1,0,0],[0,2,0],[0,0,3]] as double[][]) 
    
    assert DoubleMatrix.diag(new DoubleMatrix([1,2,3] as double[])) == diag([1,2,3], DoubleMatrix)
    
    assert diag([1,2,3]) == diag([1,2,3], DoubleMatrix)
  }
  
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

  def newMatrixColumnOrder(array, Class clazz = DoubleMatrix) {
    return clazz.newInstance(array as double[][])
  }
  
  def newMatrixRowOrder(columns, rows, getValue, Class clazz = DoubleMatrix) {
  
    def array = []
  
    for (def x = 0; x < columns; ++x) {
      array[x] = []
      for (def y = 0; y < rows; ++y) {
        def value = getValue(x, y)
        array[x][y] = value
      }
    }
  
    return newMatrixRowOrder(array, clazz)
  }

  private def newMatrixRowOrderTests() {
    assert newMatrixRowOrder([[1,2,3],[4,5,6],[7,8,9]]).toArray() == [1,2,3,4,5,6,7,8,9]
    
    assert newMatrixRowOrder([[1,2],[4,5],[7,8]]).toArray() == [1,2,4,5,7,8]
  }

  def newScaledArray(array, precision, rouding) {
    array.eachWithIndex { val, idx ->
      array[idx] = BigDecimal.valueOf(array[idx]).setScale(precision, rouding).doubleValue()
    }
  }

  def getByteBuffer(matrix) {
    def columns = 0
    def rows = 0
    def reValues = null
    def imValues = null

    def fromOapMatrix = {
      def scolumns = "columns"
      def srows = "rows"

      if (matrix.hasProperty(scolumns) && matrix.hasProperty(srows)) {
        columns = matrix."${scolumns}"
        rows = matrix."${srows}"
      }

      if (matrix.hasProperty('reValues')) {
        reValues = matrix.reValues
      }

      if (matrix.hasProperty('imValues')) {
        imValues = matrix.imValues
      }
    }

    def fromObjectArray = {
      if (matrix instanceof Object[]) {
        columns = Math.sqrt(matrix.length) as int
        rows = columns

        if (columns * rows != matrix.length) {
          throw new Exception("Square matrix can be passed as array.")
        }
        
        if (columns != 0 && rows != 0) {
          reValues = matrix
        }
      }
    }

    def from2dObjectArray = {
      if (matrix instanceof Object[][]) {
        matrix = newMatrixColumnOrder(matrix)
        reValues = matrix.toArray() 
        columns = matrix.columns
        rows = matrix.rows
      }
    }

    def fromDoubleMatrix = {
      if (matrix instanceof DoubleMatrix) {
        columns = matrix.columns
        rows = matrix.rows
        reValues = matrix.toArray()
      }
    }

    def fromList = {
      if (matrix instanceof List) {
        matrix = matrix as Object[]
        fromObjectArray(matrix)
      }
    }
    
    fromOapMatrix()
    fromObjectArray()
    from2dObjectArray()
    fromDoubleMatrix()
    fromList()

    if (reValues == null && imValues == null) {
      throw new Exception("Re and im part are null.")
    }

    def boolSize = Short.SIZE / Byte.SIZE / 1
    def intSize = Integer.SIZE / Byte.SIZE / 1
    def doubleSize = Double.SIZE / Byte.SIZE / 1

    def m = 2
    if (reValues == null) { --m; }
    if (imValues == null) { --m; }

    def bufferCapacity = intSize * 5 + boolSize * 2 + m * columns * rows * doubleSize

    def buffer = ByteBuffer.allocate(bufferCapacity as int)
    buffer.order(ByteOrder.LITTLE_ENDIAN)

    def inputStream = new PipedInputStream()
    def pipeStream = new PipedOutputStream(inputStream)
    def stream = new ObjectOutputStream(pipeStream)

    buffer.putInt(boolSize as int)
    buffer.putInt(intSize as int)
    buffer.putInt(doubleSize as int)

    buffer.putInt(columns as int)
    buffer.putInt(rows as int)

    def btos = { b ->
      return b == false ? 0 : 1
    }

    buffer.putShort(btos(reValues != null) as short)
    buffer.putShort(btos(imValues != null) as short)

    def saveCond = { section ->
      if (section != null) {
        for (def idx = 0; idx < columns * rows; ++idx) {
          buffer.putDouble(section[idx])
        }
      }
    }

    saveCond(reValues)
    saveCond(imValues)

    buffer.position(0)

    return buffer
  }

  def getBinaryData(matrix) {
    def stream = getByteBuffer(matrix)
    int length = stream.capacity()
    byte[] buffer = new byte[length]
    stream.get(buffer)
    return buffer
  }

  def saveToFile(path, data) {
    if (data instanceof String) {
      new File(dir + filename).write(data)
      return;
    }

    Files.write(Paths.get(path), data, StandardOpenOption.CREATE, StandardOpenOption.WRITE)
  }
}
