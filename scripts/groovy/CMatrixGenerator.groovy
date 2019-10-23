
def iterateMatrix (def columns, def rows, def callback) {
  def matrix = []
  for (def y = 0; y < rows; ++y) {
    matrix << []
    for (def x = 0; x < columns; ++x) {
      matrix[-1] << callback (x, y)
    }
  }
  return matrix
}

def diagonalMatrix (def columns, def rows, def callback) {
  return iterateMatrix (columns, rows) { x, y ->
    if (x == y) {
      return callback(x)
    }
    else {
      return 0
    }
  }
}

def vectorT (def columns, callback)
{
  return iterateMatrix (columns, 1) { x, y ->
    return callback (x)
  }
}

def vecToStr (def vec, def c = ["{", ",", "}"]) {
  return c[0] + vec.join(c[1]) + c[2]
}

s_matrixCppRepr =
[
  ["",",",""], ["{\n",",\n","\n}"],
]

s_vectorCppRepr =
[
  ["{",",","}"]
]

def toArrayStr (def obj) {
  def array = []
  def o = obj.collect { a ->
    if (a instanceof List) {
      return a.join(",")
    }
    return a.toString()
  }
  o = o.collect { e -> return "  " + e}
  array += o
  return array
}

assert ["  1", "  2", "  3"] == toArrayStr ([1, 2, 3])
assert ["  1,2,3"] == toArrayStr ([[1, 2, 3]])
assert ["  1", "  2", "  3"] == toArrayStr ([[1], [2], [3]])

def saveToFile (def path, def str) {
  def file = new File (path)
  file.write (str)
}

def createHeader (def testNumber)
{
  def header = []
  header << "#ifndef OAP_CMATRIXDATA${testNumber}_H"
  header << "#define OAP_CMATRIXDATA${testNumber}_H"
  header << ""
  header << "namespace CMatrixData${testNumber}"
  header << "{"
  header << ""
  return header.join("\n")
}

def createBody (def columns, def rows, def matrix, def eigenvector)
{
  def body = []
  body << "  const unsigned int columns = ${columns};"
  body << "  const unsigned int rows = ${rows};"
  body << "  double matrix[] ="
  body << "  {"
  body += toArrayStr(matrix).collect { line -> return line + ","}
  body << "  };"
  body << "  double eigenvalues[] ="
  body << "  {"
  body += toArrayStr(eigenvector).collect { line -> return line + ","}
  body << "  };"

  return body.join("\n")
}

def createTail ()
{
  def tail = []
  tail << ""
  tail << "}"
  tail << "#endif"
  return tail.join("\n")
}

def createDiagonalTest (def testNumber, def dims, def callback)
{
  def columns = dims
  def rows = dims

  def head = createHeader (testNumber);
  def tail = createTail ();

  def m = diagonalMatrix (columns, rows, callback)
  def v = vectorT (columns, callback)
  
  def body = createBody (columns, rows, m, v)

  def fileTxt = "${head}${body}${tail}"
  
  saveToFile ("/tmp/CMatrixData${testNumber}.h", fileTxt)
}

def createCMatrixTest3 ()
{
  createDiagonalTest (3, 100) {x -> (x + 1) * 3}
}

def createCMatrixTest4 ()
{
  createDiagonalTest (4, 100) {x -> x % 12 == 0 ? x : 0}
}

try {
  createCMatrixTest3 ();
  createCMatrixTest4 ();
} catch (Exception e) {
  e.printStackTrace()
}

