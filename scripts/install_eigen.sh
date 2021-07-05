#!/bin/bash
FILE="eigen-3.3.9"
ARCHIVE="$FILE.tar.gz"
EIGEN="https://gitlab.com/libeigen/eigen/-/archive/3.3.9/$ARCHIVE"
wget $EIGEN .
tar -xvf $ARCHIVE -C .
rm -r $ARCHIVE
cd $FILE
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../../3rd-party
make
make install
cd ../..
rm -r $FILE
