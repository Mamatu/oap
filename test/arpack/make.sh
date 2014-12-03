#!/bin/sh
rm -rf *.o
rm -rf main
f95 -c maxeigZF3.f  -o maxeigZF3.o
gcc -c period2Bz_MM.c -o period2Bz_MM.o
gcc maxeigZF3.o period2Bz_MM.o /usr/lib/libarpack.so /usr/lib/libparpack.so /usr/lib/libblas.so.3gf /usr/lib/liblapack.so.3gf -o main
