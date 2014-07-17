#!/bin/sh
rm -rf ./oglap
rm -rf oglap.tar.gz
mkdir oglap
 find -name '*.so' | grep 'Release' | xargs -I {} cp {} ./oglap && cp oglaShibataMgr/dist/Release/GNU-Linux-x86/oglashibatamgr ./oglap
cp oglaShibataMgr/rem ./oglap
tar -czf oglap.tar.gz ./oglap
