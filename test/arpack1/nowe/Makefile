# makefile dla periodu 2 canted
SHELL = /bin/sh

CC:=gcc
GF:=gfortran
MKLPATH= /usr/lib
FLAGS:= 
#FLAGS:= -Vaxlib -lpthread -nofor_main -fPIC -mcmodel=medium
LIBD:= -L $(MKLPATH) -Wl,--start-group /usr/lib/libarpack.so /usr/lib/libblas.so.3 /usr/lib/liblapack.so -Wl,--end-group
OBJSz:=period2BzT2.o maxeigp2.o
OBJSz2:=period2BzDMRG.o maxeigpDMRG.o

prTz: $(OBJSz)

	$(GF) -o $@ $^ $(LIBD) $(FLAGS)

maxeigp2.o: maxeigp2.f

	$(GF) -c $< -o $@

period2BzT2.o: period2BzT2.c

	$(CC) -g -c $< -o $@

prTzD: $(OBJSz2)

	$(GF) -o $@ $^ $(LIBB) $(FLAGS)

maxeigpDMRG.o: maxeigpDMRG.f

	$(GF) -g -c $< -o $@

period2BzDMRG.o: period2BzDMRG.c

	$(CC) -g -c $< -o $@

clean:

	rm -f *.o skryptFz.sh.*

.PHONY: clean
