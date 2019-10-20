package groovy

import org.jblas.*
import java.nio.*
import java.nio.file.*
import groovy.io.FileType

s_testsCount = 100
s_oapPath = ""
s_oapBinPath = "dist/Release/x86/bin/"
s_scriptFile = "/tmp/run_tests_suite.sh"
s_testLogFile = "/tmp/test_report_log.txt"

def getAllFiles (def module)
{
  def list = []
  module.eachFileRecurse (FileType.FILES)
  {
    file -> if (file.getName().contains(".cpp")) { list << file }
  }
  return list
}

def getTestsFromText (def text)
{
  def testRegex = "TEST_F\\s*\\r*(.*,.*)"
  def testRegex1 = "[a-zA-Z0-9_]*"
  def list = text.findAll(testRegex)
  list = list.collect
  {
    e -> 
    def nameList = e.toString().findAll (testRegex1)
    nameList.removeAll ("")
    return nameList[1] + "." + nameList[2]
  }

  return list
}

def getTestsFromFile (def file)
{
  return new Tuple(file.getParentFile().getName(), getTestsFromText (file.getText ()))
}

def getTestsFromModule (def path)
{
  def files = getAllFiles (path)
  def tests = []
  files.each
  {
    file -> def testsList = getTestsFromFile (file)
    tests.add (testsList)
  }
  return tests
}

def getTests (def path)
{
  def file = new File (path)
  println ("Get tests from ${path}")
  if (file.isDirectory ())
  {
    return getTestsFromModule (file)
  }
  else if (file.isFile ())
  {
    return [getTestsFromFile (file)]
  }
}

def createScript (def filepath, def tests, def commands)
{
  def contentLines = ["#!/bin/sh"]
  contentLines.addAll (commands)
  contentLines.add ("rm ${s_testLogFile}")
  tests.each
  {
    tuple ->
      def execfile = tuple.get(0)
      def list = tuple.get(1)
      list.each 
      {
        test -> s_testsCount.times { contentLines.add (s_oapBinPath + execfile + " --gtest_filter=\"" + test + "\" >> ${s_testLogFile}") }
      }
  }
  new File (filepath).write (contentLines.join("\n"))
}

def addHash (def string)
{
  if (string[-1] != "/")
  {
    return string + "/"
  }
  return string
}

def mergePathAndBinPath ()
{
  if (s_oapBinPath[0..s_oapPath.size()] != s_oapPath)
  {
    s_oapBinPath = s_oapPath + s_oapBinPath;
  }
  assert (new File(s_oapBinPath).exists ())
  assert (new File(s_oapPath).exists ())
}

s_argsProcessors =
[
  "-oap_cubin_path" : { v -> comm = "export OAP_CUBIN_PATH=${v}"; println ("-oap_cubin_path -> " + comm); return comm},
  "-tests_count" : { v -> s_testsCount = v as Integer; println ("-tests_count -> " + s_testsCount); return "#Tests count set into " + v},
  "-oap_path" : { v -> s_oapPath = addHash(v); mergePathAndBinPath (); println ("-oap_path -> " + s_oapPath); return "#Oap path is " + v},
  "-oap_bin_path" : { v -> s_oapBinPath = addHash(v); mergePathAndBinPath (); println ("-oap_bin_path -> " + s_oapBinPath); return "#Oap binaries path is " + v}
]

def filterArgs (def args, def cond)
{
  fargs = [];//args.collect()
  def keys = s_argsProcessors.keySet() as String[];
  args.each { value ->
    if (cond (value, keys)) {
      fargs << value
    }
  }
  return fargs
}

def getModules(def args)
{
  return filterArgs (args) { value, sargs ->
    for (def arg : sargs)
    {
      if (value.contains (arg))
      { return false; }
    } 
    return true;
  }
}

def getExtraArgs(def args)
{
  def extraArgs = filterArgs (args) { value, sargs ->
    for (def arg : sargs)
    {
      if (value.contains (arg)) { return true; }
    }
    return false;
  }
  def map = [:]
  extraArgs.each { value ->
    println (value)
    if (!value.contains("="))
    {
      throw new Exception ("Argument must have \"=\" to seprate key and value: ${value}")
    }
    value.splitEachLine ("=") {tokens ->
      if (tokens.size() != 2)
      {
        throw new Exception ("Arguments for this script must have: name of arguments (with \"-\" prefix) and the value. Must be splitted by \"=\"")
      }
      map[tokens[0]] = tokens[1] }
  }
  return map
}

def getCommands (args)
{
  def extraArgs = getExtraArgs (args)
  def commands = []
  extraArgs.each {
    k,v->
      def processor = s_argsProcessors[k]
      commands << processor (v)
  }
  return commands
}

try {
  assert ["CLASS.TEST"] == getTestsFromText ("TEST_F(CLASS,TEST)")
  assert ["CLASS.TEST"] == getTestsFromText ("TEST_F (CLASS,TEST)")
  assert ["CLASS.TEST"] == getTestsFromText ("TEST_F  (CLASS ,TEST)")
  assert ["CLASS.TEST"] == getTestsFromText ("TEST_F      ( CLASS  ,  TEST  )")
  
  assert getModules (["a", "-oap_cubin_path", "b"]) == ["a", "b"]
  assert getExtraArgs (["a", "-oap_cubin_path=/tmp/path", "b"]) == ["-oap_cubin_path" : "/tmp/path"]
  assert getCommands (["a", "-oap_cubin_path=/tmp/path", "b"]) == ["export OAP_CUBIN_PATH=/tmp/path"]

  def modules = getModules (args)
  def commands = getCommands (args)

  println ("Modules: " + modules)
  println ("Commands: " + commands)

  modules.each { entity ->
    def tests = getTests (s_oapPath + entity)
    createScript (s_scriptFile, tests, commands)
  }
} catch (Exception e) {
  e.printStackTrace()
}
