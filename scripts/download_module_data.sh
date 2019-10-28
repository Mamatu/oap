#!/bin/bash

DIR=../$1/data

URL=https://sourceforge.net/projects/openap/files/oap_test_data

DATE=06_02_2019
if [ "$1" == "oapNeural" ]; then
  DATE=20_10_2019
fi

ARCHIVE_NAME=oap_test_data_$1_$DATE.tar.gz

if [ "$2" == "preclean" ]; then
  rm -r $DIR/*
fi

if [ "$(ls -A $DIR)" ] ; then
  echo "Dir $DIR must be empty for this operation."
else
  wget $URL/$ARCHIVE_NAME
  tar -xvf $ARCHIVE_NAME -C .. # --strip-components=1
  rm $ARCHIVE_NAME
fi
