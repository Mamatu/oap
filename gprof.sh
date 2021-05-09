#!/bin/sh

f_help()
{
  echo '1st. - path to binary file which contains debug symbols'
  echo '2nd. - path to out file (usually it is gmon.out)'
  echo '3rd. - directory where will be created outcome of analysis'
  echo '4rd. - pattern where are contained performance data'
}

if [ $# -le 3 ] ; then
  f_help  
  exit 0
fi

DIR=$3
NAME=$4

echo "Creating $DIR/$NAME.txt"
gprof $1 $2 >> $DIR/$NAME.txt
echo "Creating $DIR/$NAME.dot"
gprof2dot $DIR/$NAME.txt > $DIR/$NAME.dot
echo "Creating $DIR/$NAME.svg"
dot -Tsvg -o $DIR/$NAME.svg $DIR/$NAME.dot
echo "Done"
