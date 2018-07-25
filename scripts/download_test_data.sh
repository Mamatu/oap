#!/bin/sh

DIR1=../oap2dt3d/data
DIR2=../oap2dt3dFuncTests/data
DIR3=../oapDeviceTests/data

URL=https://sourceforge.net/projects/openap/files/oap_test_data
DATE=08_03_2018
ARCHIVE_NAME=oap_test_data_$DATE.tar.gz

if [ "$1" == "preclean" ]; then
  rm -r $DIR1/*
  rm -r $DIR2/*
  rm -r $DIR3/*
fi

if [ "$(ls -A $DIR1)" -o "$(ls -A $DIR2)" -o "$(ls -A $DIR3)" ] ; then
  echo "Dirs oap2dt3d/data, oap2dt3dFuncTests/data, oapArnoldiDeviceTests/data must be empty for this operation."
else
  wget $URL/$ARCHIVE_NAME
  tar -xvf $ARCHIVE_NAME -C .. --strip-components=1
  rm $ARCHIVE_NAME
fi

