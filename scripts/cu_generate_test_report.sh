#!/bin/sh

groovy groovy/GenerateTestReport.groovy -oap_path=$OAP_PATH -tests_count=1 -oap_cubin_path=$OAP_PATH/dist/Release/x86/cubin oapArnoldiDeviceTests/oapArnoldiPackageCallbackTests.cpp oapDeviceTests oapAppUtilsFuncTests oap2dt3dDeviceTests
sh /tmp/run_tests_suite.sh

groovy groovy/AnalysisTestReport.groovy
