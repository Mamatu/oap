#!/bin/sh

DIR1=../oap2dt3d/data
DIR2=../oap2dt3dFuncTests/data
DIR3=../oapDeviceTests/data

URL=https://sourceforge.net/projects/openap/files/oap_test_data
ARCHIVE_NAME=oap_test_data.tar.gz

if [ "$1" == "preclean" ]; then
  rm -r $DIR1/*
  rm -r $DIR2/*
  rm -r $DIR3/*
fi

if [ "$(ls -A $DIR1)" -o "$(ls -A $DIR2)" -o "$(ls -A $DIR3)" ] ; then
  echo "Dirs oap2dt3d/data, oap2dt3dFuncTests/data, oapDeviceTests/data must be empty for this operation."
else
  wget $URL/$ARCHIVE_NAME
  tar -xvf $ARCHIVE_NAME -C .. --strip-components=2
  rm $ARCHIVE_NAME
fi

