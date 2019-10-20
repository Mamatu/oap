def matmul(m,n, printCoords = null)
{
  def h = m.size()
  def w = n[0].size()
  def ll = n.size()
  assert (ll == m[0].size())
  output = [] as ArrayList
  for (def y = 0; y < h; ++y)
  {
    def row = [] as ArrayList
    for (def x = 0; x < w; ++x)
    {
      def value = 0
      for (def z = 0; z < ll; ++z)
      {
        def pvalue = value
        def v1 = m[y][z]
        def v2 = n[z][x]
        value += v1 * v2
        if (printCoords != null)
        {
          if (printCoords.contains([x,y]))
          {
            println "${value} (${x}, ${y}) = ${pvalue} + ${v1 * v2} ( = ${v1} * ${v2})"
          }
        }
      }
      row.add(value)
    }
    output.add (row)
  }
  return output
}

def printMatrix (_str, m, matrixClosure, rowClosure, joinStr, format = null)
{
  str = "${_str}"
  str += "${matrixClosure[0]}"

  def array1 = []
  for (def x = 0; x < m.size(); ++x)
  {
    def row = m[x]
    if (format != null)
    {
      def array2 = []
      m[x].each { v -> 
        float value = (float)v
        def prefix = ""
        if (value >= 0)
        { 
          prefix = "+"
        }
        array2.add(prefix + String.format(format, value)) }
      row = array2
    }
    array1.add (row.join(joinStr))
  }

  str += "${rowClosure[0]}" + array1.join ("${rowClosure[1]}${rowClosure[0]}")
  str += "${rowClosure[2]}"
  str += "${matrixClosure[1]}"

  println str
}

def printMatrix (_str, m)
{
  printMatrix (_str, m, ["[\n","]\n"], ["[","]\n", "]\n"], ", ", "%.9f")
}

def printMatrix (m)
{
  printMatrix ("", m)
}


def unwanted = 3.0010e+00
def M =
[
[-4.4529e-01 - unwanted, -1.8641e+00, -2.8109e+00, 7.2941e+00],
[8.0124e+00, 6.2898e+00 - unwanted, 1.2058e+01, -1.6088e+01],
[0.0000e+00, 4.0087e-01, 1.1545e+00 - unwanted, -3.3722e-01],
[0.0000e+00, 0.0000e+00, -1.5744e-01, 3.0010e+00 - unwanted],
]

printMatrix ("Init = \n", M, ["", "\n"], ["", "\n", ""], " ")

def runGivensRotation (M, column, row, expectedA, expectedB)
{
  println "Givens Rotation ${column} ${row}"
  def A = M[column][column]
  if (expectedA != null)
  {
    assert (A == expectedA)
  }
  def B = M[row][column]

  if (expectedB != null)
  {
    assert (B == expectedB)
  }

  def R = Math.hypot (A, B)
  def C = A / R
  def S = -B / R

  println "A ${A}"
  println "B ${B}"
  println "C ${C}"
  println "S ${S}"

  def G =
  [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  ]

  G[column][column] = C
  G[row][row] = C
  G[column][row] = -S
  G[row][column] = S

  def O = matmul (G, M, [[2, 2],[3, 2]])

  //printMatrix ("O", O)
  printMatrix ("G", G)
  //printMatrix ("M", M)
  //println ""
  return O
}

def getColumnsCount (m)
{
  return m[0].size()
}

def getRowsCount (m)
{
  return m.size()
}

def qrDecomposition (AM)
{
  def M = AM.collect { return it.clone() }
  def columnsCount = getColumnsCount (M)
  def rowsCount = getRowsCount (M)

  for (def j = 0; j < columnsCount; ++j)
  {
    for (def i = j + 1; i < rowsCount; ++i)
    {
      if (M[i][j] != 0)
      {
        def r = Math.sqrt (Math.pow (M[j][j], 2) + Math.pow (M[i][j], 2))
        if (M[i][j] < 0)
        {
          r = -r
        }
        def s = M[i][j] / r
        def c = M[j][j] / r

        for (def k = j; k < columnsCount; ++k)
        {
          def jk = M[j][k]
          def ik = M[i][k]
          M[j][k] = c * jk + s * ik
          M[i][k] = -s * jk + c * ik
        }
        
        if (c == 0)
        {
          M[i][j] = 1
        }
        else if (Math.abs (s) < Math.abs (c))
        {
          if (c < 0)
          {
            M[i][j] = -0.5 * s
          }
          else
          {
            M[i][j] = 0.5 * s
          }
        }
        else
        {
          M[i][j] = (float)2 / c
        }
      }
    }
  }
  return M
}

def getR (AM)
{
  def R = AM.collect { return it.clone() }
  def rowsCount = getRowsCount (R)
  def columnsCount = getColumnsCount (R)
  for (def i = 0; i < rowsCount; ++i)
  {
    for (def j = 0; j < i; ++j)
    {
      R[i][j] = 0
    }
  }
  return R
}

MO = getR (qrDecomposition (M))
printMatrix ("MO = ", MO)

def runGivensRotation (M, column, row)
{
  return runGivensRotation (M, column, row, null, null)
}

crs = [[0, 1, [-4.4529e-01 - unwanted, 8.0124e+00]], [0, 2], [1, 2], [2, 3]]
//crs = [[2, 3], [1, 2], [0, 1]]

printMatrix("M", M)

crs.each { coords ->
  def ev1 = null
  def ev2 = null
  if (coords.size() == 3)
  {
    ev1 = coords[2][0]
    ev2 = coords[2][1]
  }
  M = runGivensRotation (M, coords[0], coords[1], ev1, ev2)
  printMatrix("M", M)
}

def Expected =
[
[8.722,   3.758,  12.187, -17.661],
[0.000,   0.576,  -2.852,  -0.482],
[0.000,   0.000,  -0.251,   0.002],
[0.000,   0.000,   0.000,  -0.002],
]

def calcDiff (m1, m2)
{
  def o = []
  assert (m1.size() == m2.size())
  for (def x = 0; x < m1.size(); ++x)
  {
    def row = []
    assert (m1[x].size() == m2[x].size())
    for (def y = 0; y < m1[x].size(); ++y)
    {
      def v = m1[x][y] - m2[x][y]
      row.add (v)
    }
    o.add (row)
  }
  return o
}

def printInfo (expected, matrix)
{
  def pm = {s, m -> printMatrix (s, m, ["[","]"], ["", ", ", ""], ", ", "%.9f")}
  pm ("Output   = ", matrix)
  pm ("Expected = ", expected)
  pm ("Diff     = ", calcDiff (matrix, expected))
}

printInfo (Expected, MO)
