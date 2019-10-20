package groovy

import org.jblas.*
import java.nio.*
import java.nio.file.*
import groovy.io.FileType

s_testLogFile = "/tmp/test_report_log.txt"
s_FAILED = "FAILED"
s_PASSED = "OK"
def parseLog (def content)
{
  def testRegex = "\\[\\s*?(${s_FAILED}|${s_PASSED})\\s*?\\] ([a-zA-Z0-9_.]*) \\(([0-9]*) ms\\)"
  def testsMap = [:]
  content.findAll (testRegex) {
    matcher, status, name, time ->
      if (!testsMap.containsKey (name)) {
        testsMap[name] = new Tuple (name, [] as ArrayList, [] as ArrayList)
      }
      testsMap[name].get(1).add (status)
      testsMap[name].get(2).add (time as Integer)
  }
  return testsMap
}

def getMean (def timesList)
{
  def value = 0
  timesList.each { t ->
    value += t
  }
  return value / timesList.size();
}

def checkStatus (def statusList)
{
  def sl = statusList.toUnique ()
  if (sl.size() != 1)
  {
    throw new Exception ("not supported: not one status")
  }
  return sl[0]
}

def printLog (def tests)
{
  tests.each { k, v ->
    def time = getMean (v.get (2))
    println ("${checkStatus (v.get(1))}: ${v.get(0)} ${time} ms")
  }
}

try {
  assert ["A.B" : ["A.B", ["OK"], [10]]] == parseLog ("[   OK  ] A.B (10 ms)")
  def file = new File (s_testLogFile)
  def content = file.getText ()

  def tests = parseLog (content)
  printLog (tests)
} catch (Exception e) {
  e.printStackTrace()
}
